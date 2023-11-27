import gc
import logging
from dataclasses import asdict, astuple, dataclass
from time import perf_counter
from typing import List, NamedTuple, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.rollout.rollout import RolloutGenerator
from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.autocast import maybe_autocast
from rl_algo_impls.shared.callbacks import Callback
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.schedule import update_learning_rate
from rl_algo_impls.shared.stats import log_scalars
from rl_algo_impls.shared.tensor_utils import NumOrList, num_or_array


class TrainStepStats(NamedTuple):
    loss: float
    pi_loss: float
    v_loss: np.ndarray
    entropy_loss: float
    approx_kl: float
    clipped_frac: float
    val_clipped_frac: np.ndarray


@dataclass
class TrainStats:
    loss: float
    pi_loss: float
    v_loss: Union[float, np.ndarray]
    entropy_loss: float
    approx_kl: float
    clipped_frac: float
    val_clipped_frac: Union[float, np.ndarray]
    explained_var: float

    def __init__(self, step_stats: List[TrainStepStats], explained_var: float) -> None:
        self.loss = np.mean([s.loss for s in step_stats]).item()
        self.pi_loss = np.mean([s.pi_loss for s in step_stats]).item()
        self.v_loss = np.mean([s.v_loss for s in step_stats], axis=0)
        self.entropy_loss = np.mean([s.entropy_loss for s in step_stats]).item()
        self.approx_kl = np.mean([s.approx_kl for s in step_stats]).item()
        self.clipped_frac = np.mean([s.clipped_frac for s in step_stats]).item()
        self.val_clipped_frac = np.mean(
            [s.val_clipped_frac for s in step_stats], axis=0
        )
        self.explained_var = explained_var

    def write_to_tensorboard(self, tb_writer: SummaryWriter, global_step: int) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                for idx, v in enumerate(value.flatten()):
                    tb_writer.add_scalar(
                        f"losses/{name}_{idx}", v, global_step=global_step
                    )
            else:
                tb_writer.add_scalar(f"losses/{name}", value, global_step=global_step)

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
NL = TypeVar("NL", float, List[float])


class PPO(Algorithm):
    def __init__(
        self,
        policy: ActorCritic,
        device: torch.device,
        tb_writer: SummaryWriter,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: NL = 0.99,
        gae_lambda: NumOrList = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        normalize_advantage: bool = True,
        standardize_advantage: bool = False,
        ent_coef: float = 0.0,
        vf_coef: NumOrList = 0.5,
        ppo2_vf_coef_halving: bool = False,
        max_grad_norm: float = 0.5,
        multi_reward_weights: Optional[List[int]] = None,
        gradient_accumulation: bool = False,
        kl_cutoff: Optional[float] = None,
        freeze_policy_head: bool = False,
        freeze_value_head: bool = False,
        freeze_backbone: bool = False,
        switch_range: Optional[int] = None,
        guide_probability: Optional[float] = None,
        normalize_advantages_after_scaling: bool = False,
        autocast_loss: bool = False,
    ) -> None:
        super().__init__(policy, device, tb_writer)
        self.policy = policy

        self.gamma = num_or_array(gamma)
        self.gae_lambda = num_or_array(gae_lambda)
        self.optimizer = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-7)
        self.learning_rate = learning_rate
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

        self.switch_range = switch_range
        self.guide_probability = guide_probability
        self.normalize_advantages_after_scaling = normalize_advantages_after_scaling

        self.autocast_loss = autocast_loss

    def learn(
        self: PPOSelf,
        train_timesteps: int,
        rollout_generator: RolloutGenerator,
        callbacks: Optional[List[Callback]] = None,
        total_timesteps: Optional[int] = None,
        start_timesteps: int = 0,
    ) -> PPOSelf:
        if total_timesteps is None:
            total_timesteps = train_timesteps
        assert start_timesteps + train_timesteps <= total_timesteps

        timesteps_elapsed = start_timesteps
        while timesteps_elapsed < start_timesteps + train_timesteps:
            timesteps_elapsed, should_continue = self.learn_epoch(
                timesteps_elapsed, total_timesteps, rollout_generator, callbacks
            )
            gc.collect()
            if not should_continue:
                break
        return self

    def learn_epoch(
        self,
        timesteps_elapsed: int,
        total_timesteps: int,
        rollout_generator: RolloutGenerator,
        callbacks: Optional[List[Callback]] = None,
    ) -> Tuple[int, bool]:
        start_time = perf_counter()

        update_learning_rate(self.optimizer, self.learning_rate)
        pi_clip = self.clip_range
        chart_scalars = {
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "ent_coef": self.ent_coef,
            "pi_clip": pi_clip,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "vf_coef": self.vf_coef,
        }
        if self.clip_range_vf is not None:
            v_clip = self.clip_range_vf
            chart_scalars["v_clip"] = v_clip
        else:
            v_clip = None
        if self.multi_reward_weights is not None:
            chart_scalars["reward_weights"] = self.multi_reward_weights
        if self.switch_range is not None:
            assert hasattr(
                rollout_generator, "switch_range"
            ), f"rollout_generator assumed to have switch_range attribute"
            setattr(rollout_generator, "switch_range", self.switch_range)
            chart_scalars["switch_range"] = self.switch_range
        if self.guide_probability is not None:
            assert hasattr(
                rollout_generator, "guide_probability"
            ), f"rollout_generator assumed to have guide_probability attribute"
            setattr(rollout_generator, "guide_probability", self.guide_probability)
            chart_scalars["guide_probability"] = self.guide_probability
        log_scalars(self.tb_writer, "charts", chart_scalars, timesteps_elapsed)

        r = rollout_generator.rollout(gamma=self.gamma, gae_lambda=self.gae_lambda)
        gc.collect()
        timesteps_elapsed += r.total_steps

        step_stats = []
        multi_reward_weights = (
            torch.Tensor(self.multi_reward_weights).to(self.device)
            if self.multi_reward_weights is not None
            else None
        )
        vf_coef = torch.Tensor(np.array(self.vf_coef)).to(self.device)
        pi_coef = 1
        if self.freeze_policy_head or self.freeze_value_head or self.freeze_backbone:
            self.policy.freeze(
                self.freeze_policy_head,
                self.freeze_value_head,
                self.freeze_backbone,
            )
        for _ in range(self.n_epochs):
            # Only record last epoch's stats
            step_stats.clear()
            for mb in r.minibatches(self.batch_size, self.device):
                self.policy.reset_noise(self.batch_size)

                (
                    mb_obs,
                    mb_logprobs,
                    mb_actions,
                    mb_action_masks,
                    _,
                    mb_values,
                    mb_adv,
                    mb_returns,
                ) = astuple(mb)

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

                with maybe_autocast(self.autocast_loss, self.device):
                    new_logprobs, entropy, new_values = self.policy(
                        mb_obs, mb_actions, action_masks=mb_action_masks
                    )

                    logratio = new_logprobs - mb_logprobs
                    ratio = torch.exp(logratio)
                    clipped_ratio = torch.clamp(ratio, min=1 - pi_clip, max=1 + pi_clip)
                    pi_loss = -torch.min(ratio * mb_adv, clipped_ratio * mb_adv).mean()

                    v_loss_unclipped = (new_values - mb_returns) ** 2
                    if v_clip is not None:
                        v_loss_clipped = (
                            mb_values
                            + torch.clamp(new_values - mb_values, -v_clip, v_clip)
                            - mb_returns
                        ) ** 2
                        v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean(0)
                    else:
                        v_loss = v_loss_unclipped.mean(0)

                    if self.ppo2_vf_coef_halving:
                        v_loss *= 0.5

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

                    if self.gradient_accumulation:
                        loss /= r.num_minibatches(self.batch_size)
                loss.backward()
                if not self.gradient_accumulation:
                    self.optimizer_step()

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
                        v_loss.detach().cpu().numpy(),
                        entropy_loss.item(),
                        approx_kl,
                        clipped_frac,
                        val_clipped_frac,
                    )
                )
            if self.gradient_accumulation:
                self.optimizer_step()
        if self.freeze_policy_head or self.freeze_value_head or self.freeze_backbone:
            self.policy.unfreeze()

        var_y = np.var(r.y_true).item()
        explained_var = (
            np.nan if var_y == 0 else 1 - np.var(r.y_true - r.y_pred).item() / var_y
        )
        train_stats = TrainStats(step_stats, explained_var)
        train_stats.write_to_tensorboard(self.tb_writer, timesteps_elapsed)

        end_time = perf_counter()
        rollout_steps = r.total_steps
        self.tb_writer.add_scalar(
            "train/steps_per_second",
            rollout_steps / (end_time - start_time),
            timesteps_elapsed,
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

    def optimizer_step(self) -> None:
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
