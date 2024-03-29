import gc
import logging
import warnings
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Dict, List, NamedTuple, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from rl_algo_impls.loss.teacher_kl_loss import TeacherKLLoss
from rl_algo_impls.rollout.ppo_rollout import PPOBatch
from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.autocast import maybe_autocast
from rl_algo_impls.shared.callbacks import Callback
from rl_algo_impls.shared.data_store.data_store_data import LearnerDataStoreViewUpdate
from rl_algo_impls.shared.data_store.data_store_view import LearnerDataStoreView
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.schedule import update_learning_rate
from rl_algo_impls.shared.stats import log_scalars
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.shared.tensor_utils import NumOrList, num_or_array


class TrainStepStats(NamedTuple):
    loss: float
    pi_loss: float
    v_loss: np.ndarray
    entropy_loss: float
    approx_kl: float
    clipped_frac: float
    val_clipped_frac: np.ndarray
    additional_losses: Dict[str, float]


@dataclass
class TrainStats:
    loss: float
    pi_loss: float
    v_loss: Union[float, np.ndarray]
    entropy_loss: float
    approx_kl: float
    clipped_frac: float
    val_clipped_frac: Union[float, np.ndarray]
    additional_losses: Dict[str, float]
    explained_var: float
    grad_norm: float

    def __init__(
        self,
        step_stats: List[TrainStepStats],
        explained_var: float,
        grad_norms: List[float],
    ) -> None:
        self.loss = np.mean([s.loss for s in step_stats]).item()
        self.pi_loss = np.mean([s.pi_loss for s in step_stats]).item()
        self.v_loss = np.mean([s.v_loss for s in step_stats], axis=0)
        self.entropy_loss = np.mean([s.entropy_loss for s in step_stats]).item()
        self.approx_kl = np.mean([s.approx_kl for s in step_stats]).item()
        self.clipped_frac = np.mean([s.clipped_frac for s in step_stats]).item()
        self.val_clipped_frac = np.mean(
            [s.val_clipped_frac for s in step_stats], axis=0
        )
        self.additional_losses = {
            k: np.mean([s.additional_losses[k] for s in step_stats]).item()
            for k in step_stats[0].additional_losses
        }
        self.explained_var = explained_var
        self.grad_norm = np.mean(grad_norms).item()

    def write_to_tensorboard(self, tb_writer: AbstractSummaryWrapper) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                for idx, v in enumerate(value.flatten()):
                    tb_writer.add_scalar(f"losses/{name}_{idx}", v)
            elif isinstance(value, dict):
                for k, v in value.items():
                    tb_writer.add_scalar(f"losses/{k}", v)
            else:
                tb_writer.add_scalar(f"losses/{name}", value)

    def __repr__(self) -> str:
        def round_list_or_float(v: Union[float, np.ndarray], ndigits: int) -> str:
            if isinstance(v, np.ndarray):
                return "[" + ", ".join(round(a, ndigits) for a in v) + "]"
            else:
                return str(round(v, ndigits))

        return " | ".join(
            [
                f"Loss: {round(self.loss, 2)}",
                f"Pi L: {round(self.pi_loss, 2)}",
                f"V L: {round_list_or_float(self.v_loss, 2)}",
                f"E L: {round(self.entropy_loss, 2)}",
                f"Apx KL Div: {round(self.approx_kl, 2)}",
                f"Clip Frac: {round(self.clipped_frac, 2)}",
                f"Val Clip Frac: {round_list_or_float(self.val_clipped_frac, 2)}",
            ]
        )


PPOSelf = TypeVar("PPOSelf", bound="PPO")


class PPO(Algorithm):
    def __init__(
        self,
        policy: ActorCritic,
        device: torch.device,
        tb_writer: AbstractSummaryWrapper,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        n_epochs: int = 10,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        normalize_advantage: bool = True,
        standardize_advantage: bool = False,
        ent_coef: float = 0.0,
        vf_coef: NumOrList = 0.5,
        ppo2_vf_coef_halving: bool = False,
        max_grad_norm: float = 0.5,
        multi_reward_weights: Optional[List[int]] = None,
        gradient_accumulation: Union[bool, int] = False,
        kl_cutoff: Optional[float] = None,
        freeze_policy_head: bool = False,
        freeze_value_head: bool = False,
        freeze_backbone: bool = False,
        normalize_advantages_after_scaling: bool = False,
        autocast_loss: bool = False,
        vf_loss_fn: str = "mse_loss",
        vf_weights: Optional[List[int]] = None,
        teacher_kl_loss_coef: Optional[float] = None,
        teacher_loss_importance_sampling: bool = True,
        scale_loss_by_has_actions: bool = False,
        scale_loss_by_num_actions: bool = False,
        optim_eps: float = 1e-7,
        optim_weight_decay: float = 0.0,
        optim_betas: Optional[List[int]] = None,
        optim_amsgrad: bool = False,
    ) -> None:
        super().__init__(
            policy,
            device,
            tb_writer,
            learning_rate,
            AdamW(
                policy.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999) if optim_betas is None else tuple(optim_betas),  # type: ignore
                eps=optim_eps,
                weight_decay=optim_weight_decay,
                amsgrad=optim_amsgrad,
            ),
        )
        self.policy = policy

        self.max_grad_norm = max_grad_norm
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf

        self.normalize_advantage = normalize_advantage
        self.standardize_advantage = standardize_advantage
        assert not (
            normalize_advantage and standardize_advantage
        ), f"Cannot both normalize and standardize advantage"

        self.ent_coef = ent_coef
        self.vf_coef = num_or_array(vf_coef)
        self.vf_weights = np.array(vf_weights) if vf_weights is not None else None
        self.ppo2_vf_coef_halving = ppo2_vf_coef_halving

        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.multi_reward_weights = (
            np.array(multi_reward_weights) if multi_reward_weights else None
        )
        self.gradient_accumulation = gradient_accumulation
        self.kl_cutoff = kl_cutoff

        self.freeze_policy_head = freeze_policy_head
        self.freeze_value_head = freeze_value_head
        self.freeze_backbone = freeze_backbone

        self.normalize_advantages_after_scaling = normalize_advantages_after_scaling

        self.autocast_loss = autocast_loss

        self.vf_loss_fn = getattr(F, vf_loss_fn)

        self.teacher_kl_loss_coef = teacher_kl_loss_coef
        self.teacher_kl_loss_fn = TeacherKLLoss() if teacher_kl_loss_coef else None
        self.teacher_loss_importance_sampling = teacher_loss_importance_sampling

        assert (
            bool(scale_loss_by_has_actions) + bool(scale_loss_by_num_actions) <= 1
        ), f"Cannot scale loss by both has_actions and num_actions"
        self.scale_loss_by_has_actions = scale_loss_by_has_actions
        self.scale_loss_by_num_actions = scale_loss_by_num_actions

    def learn(
        self: PPOSelf,
        learner_data_store_view: LearnerDataStoreView,
        train_timesteps: int,
        callbacks: Optional[List[Callback]] = None,
    ) -> PPOSelf:
        timesteps_elapsed = 0
        learner_data_store_view.submit_learner_update(
            LearnerDataStoreViewUpdate(self.policy, self, timesteps_elapsed)
        )
        while timesteps_elapsed < train_timesteps:
            timesteps_elapsed, should_continue = self.learn_epoch(
                learner_data_store_view, timesteps_elapsed, callbacks
            )
            learner_data_store_view.submit_learner_update(
                LearnerDataStoreViewUpdate(self.policy, self, timesteps_elapsed)
            )
            gc.collect()
            if not should_continue:
                break
        return self

    def learn_epoch(
        self,
        learner_data_store_view: LearnerDataStoreView,
        timesteps_elapsed: int,
        callbacks: Optional[List[Callback]] = None,
    ) -> Tuple[int, bool]:
        start_time = perf_counter()

        update_learning_rate(self.optimizer, self.learning_rate)
        pi_clip = self.clip_range
        chart_scalars = {
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "ent_coef": self.ent_coef,
            "pi_clip": pi_clip,
            "vf_coef": self.vf_coef,
        }
        if self.clip_range_vf is not None:
            v_clip = self.clip_range_vf
            chart_scalars["v_clip"] = v_clip
        else:
            v_clip = None
        if self.multi_reward_weights is not None:
            chart_scalars["reward_weights"] = self.multi_reward_weights
        if self.vf_weights is not None:
            chart_scalars["vf_weights"] = self.vf_weights
        if self.teacher_kl_loss_coef is not None:
            chart_scalars["teacher_kl_loss_coef"] = self.teacher_kl_loss_coef
        log_scalars(self.tb_writer, "charts", chart_scalars)

        (rollouts,) = learner_data_store_view.get_learner_view()
        if len(rollouts) > 1:
            warnings.warn(
                f"PPO does not support multiple rollouts ({len(rollouts)}) per epoch. "
                "Only the last rollout will be used"
            )
        r = rollouts[-1]

        gc.collect()
        timesteps_elapsed += r.total_steps

        step_stats = []
        grad_norms = []
        multi_reward_weights = (
            torch.Tensor(self.multi_reward_weights).to(self.device)
            if self.multi_reward_weights is not None
            else None
        )
        vf_coef = torch.Tensor(np.array(self.vf_coef)).to(self.device)
        vf_weights = (
            torch.Tensor(self.vf_weights).to(self.device)
            if self.vf_weights is not None
            else None
        )
        pi_coef = 1
        if self.freeze_policy_head or self.freeze_value_head or self.freeze_backbone:
            self.policy.freeze(
                self.freeze_policy_head,
                self.freeze_value_head,
                self.freeze_backbone,
            )
        shuffle_minibatches = True
        if self.gradient_accumulation:
            if self.gradient_accumulation is True:
                minibatches_per_step = r.num_minibatches(self.batch_size)
                shuffle_minibatches = False
            else:
                minibatches_per_step = self.gradient_accumulation
        else:
            minibatches_per_step = 1
        for _ in range(self.n_epochs):
            # Only record last epoch's stats
            step_stats.clear()
            grad_norms.clear()
            mb_idx = 0
            for mb in r.minibatches(
                self.batch_size, self.device, shuffle=shuffle_minibatches
            ):
                mb_idx += 1
                self.policy.reset_noise(self.batch_size)

                assert isinstance(mb, PPOBatch)
                (
                    mb_obs,
                    mb_actions,
                    mb_action_masks,
                    mb_logprobs,
                    mb_values,
                    mb_adv,
                    mb_returns,
                    mb_teacher_logprobs,
                    mb_num_actions,
                ) = mb

                if self.normalize_advantages_after_scaling:
                    if multi_reward_weights is not None:
                        mb_adv = mb_adv @ multi_reward_weights

                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                else:
                    if self.normalize_advantage:
                        mb_adv = (mb_adv - mb_adv.mean(0)) / (mb_adv.std(0) + 1e-8)
                    elif self.standardize_advantage:
                        mb_adv = mb_adv / (mb_adv.std(0) + 1e-8)
                    if multi_reward_weights is not None:
                        mb_adv = mb_adv @ multi_reward_weights

                loss_factor = None
                if self.scale_loss_by_has_actions:
                    assert mb_num_actions is not None
                    mb_has_actions = mb_num_actions.bool().float()
                    loss_factor = mb_has_actions / mb_has_actions.sum()
                elif self.scale_loss_by_num_actions:
                    assert mb_num_actions is not None
                    loss_factor = mb_num_actions.float() / mb_num_actions.sum()

                additional_losses = {}
                with maybe_autocast(self.autocast_loss, self.device):
                    new_logprobs, entropy, new_values = self.policy(
                        mb_obs, mb_actions, action_masks=mb_action_masks
                    )

                    logratio = new_logprobs - mb_logprobs
                    ratio = torch.exp(logratio)
                    clipped_ratio = torch.clamp(ratio, min=1 - pi_clip, max=1 + pi_clip)
                    pi_loss = -torch.min(ratio * mb_adv, clipped_ratio * mb_adv)
                    if loss_factor is not None:
                        pi_loss = (pi_loss * loss_factor).sum()
                    else:
                        pi_loss = pi_loss.mean()

                    v_loss_unclipped = self.vf_loss_fn(
                        new_values, mb_returns, reduction="none"
                    )
                    if v_clip is not None:
                        v_loss_clipped = self.vf_loss_fn(
                            mb_values
                            + torch.clamp(new_values - mb_values, -v_clip, v_clip),
                            mb_returns,
                            reduction="none",
                        )
                        v_loss = torch.max(v_loss_unclipped, v_loss_clipped)
                    else:
                        v_loss = v_loss_unclipped
                    if vf_weights is not None:
                        v_loss = v_loss @ vf_weights
                    v_loss = v_loss.mean(0)

                    if self.ppo2_vf_coef_halving:
                        v_loss *= 0.5

                    if loss_factor is not None:
                        entropy_loss = -(entropy * loss_factor).sum()
                    else:
                        entropy_loss = -entropy.mean()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean().cpu().numpy().item()
                    if self.kl_cutoff is not None and approx_kl > self.kl_cutoff:
                        pi_coef = 0

                    loss = (
                        pi_coef * pi_loss
                        + self.ent_coef * entropy_loss
                        + (vf_coef * v_loss).sum()
                    )

                    if self.teacher_kl_loss_coef:
                        assert self.teacher_kl_loss_fn
                        assert (
                            mb_teacher_logprobs is not None
                        ), "Teacher logprobs missing"
                        teacher_kl_loss = self.teacher_kl_loss_fn(
                            new_logprobs,
                            mb_teacher_logprobs,
                            ratio if self.teacher_loss_importance_sampling else None,
                        )
                        additional_losses["teacher_kl_loss"] = teacher_kl_loss.item()
                        loss += self.teacher_kl_loss_coef * teacher_kl_loss

                    loss /= minibatches_per_step
                loss.backward()
                if mb_idx % minibatches_per_step == 0:
                    grad_norms.append(self.optimizer_step())

                with torch.no_grad():
                    clipped_frac = (
                        ((ratio - 1).abs() > pi_clip)
                        .float()
                        .mean()
                        .cpu()
                        .numpy()
                        .item()
                    )
                    val_clipped_frac = (
                        ((new_values - mb_values).abs() > v_clip)
                        .float()
                        .mean(0)
                        .cpu()
                        .numpy()
                        if v_clip is not None
                        else np.zeros(v_loss.shape)
                    )

                step_stats.append(
                    TrainStepStats(
                        loss.item(),
                        pi_loss.item(),
                        v_loss.detach().float().cpu().numpy(),
                        entropy_loss.item(),
                        approx_kl,
                        clipped_frac,
                        val_clipped_frac,
                        additional_losses,
                    )
                )
            if mb_idx % minibatches_per_step != 0:
                grad_norms.append(self.optimizer_step())
        if self.freeze_policy_head or self.freeze_value_head or self.freeze_backbone:
            self.policy.unfreeze()

        self.tb_writer.on_timesteps_elapsed(timesteps_elapsed)

        var_y = np.var(r.y_true).item()
        explained_var = (
            np.nan if var_y == 0 else 1 - np.var(r.y_true - r.y_pred).item() / var_y
        )

        train_stats = TrainStats(step_stats, explained_var, grad_norms)
        train_stats.write_to_tensorboard(self.tb_writer)

        end_time = perf_counter()
        rollout_steps = r.total_steps
        self.tb_writer.add_scalar(
            "train/steps_per_second",
            rollout_steps / (end_time - start_time),
        )

        if callbacks:
            if not all(
                c.on_step(timesteps_elapsed=rollout_steps, train_stats=train_stats)
                for c in callbacks
            ):
                logging.info(
                    f"Callback terminated training at {timesteps_elapsed} timesteps"
                )
                return timesteps_elapsed, False
        return timesteps_elapsed, True

    def optimizer_step(self) -> float:
        grad_norm = nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm
        ).item()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm
