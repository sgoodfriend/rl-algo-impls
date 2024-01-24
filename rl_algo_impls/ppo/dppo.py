import gc
import logging
from time import perf_counter
from typing import List, Optional, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from rl_algo_impls.loss.teacher_kl_loss import TeacherKLLoss
from rl_algo_impls.ppo.dppo_train_stats import DPPOTrainStats, DPPOTrainStepStats
from rl_algo_impls.rollout.rollout_dataloader import RolloutDataLoader
from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.callbacks.callback import Callback
from rl_algo_impls.shared.data_store.data_store_data import LearnerDataStoreViewUpdate
from rl_algo_impls.shared.data_store.data_store_view import LearnerDataStoreView
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.schedule import SetLRScheduler
from rl_algo_impls.shared.stats import log_scalars
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.shared.tensor_utils import NumOrList, num_or_array

DPPOSelf = TypeVar("DPPOSelf", bound="DPPO")


class DPPO(Algorithm):
    def __init__(
        self,
        policy: ActorCritic,
        device: torch.device,
        tb_writer: AbstractSummaryWrapper,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
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
        teacher_loss_batch_size: Optional[int] = None,
        max_n_epochs: Optional[int] = 10,
    ) -> None:
        super().__init__(
            policy,
            device,
            tb_writer,
            learning_rate,
            Adam(policy.parameters(), lr=learning_rate, eps=1e-7),
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
        self.teacher_loss_batch_size = teacher_loss_batch_size

        self.max_n_epochs = max_n_epochs

    def learn(
        self: DPPOSelf,
        learner_data_store_view: LearnerDataStoreView,
        train_timesteps: int,
        callbacks: Optional[List[Callback]] = None,
    ) -> DPPOSelf:
        from accelerate import Accelerator

        timesteps_elapsed = 0
        learner_data_store_view.submit_learner_update(
            LearnerDataStoreViewUpdate(self.policy, self, timesteps_elapsed)
        )
        (rollouts,) = learner_data_store_view.get_learner_view(wait=True)

        shuffle_minibatches = True
        if self.gradient_accumulation:
            if self.gradient_accumulation is True:
                n_learners = torch.cuda.device_count() or 1
                minibatches_per_step = (
                    rollouts[0].num_minibatches(self.batch_size) // n_learners
                )
                shuffle_minibatches = False
            else:
                minibatches_per_step = self.gradient_accumulation
        else:
            minibatches_per_step = 1

        lr_scheduler = SetLRScheduler(self.optimizer)

        accelerator = Accelerator(
            gradient_accumulation_steps=minibatches_per_step,
            mixed_precision="bf16"
            if self.autocast_loss
            and self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
            else None,
            device_placement=self.device.type != "cpu",
            cpu=self.device.type == "cpu",
        )
        logging.info(
            f"Accelerator num_processed: {accelerator.num_processes}; "
            f"distributed_type: {accelerator.distributed_type}; "
            f"mixed_precision: {accelerator.mixed_precision}; "
            f"use_distributed": {accelerator.use_distributed}"
        )
        policy, optimizer, lr_scheduler = accelerator.prepare(
            self.policy, self.optimizer, lr_scheduler
        )

        while timesteps_elapsed < train_timesteps:
            start_time = perf_counter()

            lr_scheduler.step(self.learning_rate)
            pi_clip = self.clip_range
            chart_scalars = {
                "learning_rate": self.learning_rate,
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

            if (
                self.freeze_policy_head
                or self.freeze_value_head
                or self.freeze_backbone
            ):
                policy.freeze(
                    self.freeze_policy_head,
                    self.freeze_value_head,
                    self.freeze_backbone,
                )

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

            next_rollouts = tuple()
            total_rollout_steps = sum(r.total_steps for r in rollouts)
            rollout_iteration_cnt = 0
            rollout_steps_elapsed = 0
            rollout_steps_iteration = 0
            n_epochs = 0
            step_stats = []
            y_true_list = []
            y_pred_list = []
            while not next_rollouts:
                rollout_iteration_cnt += 1
                rollout_steps_iteration = 0
                step_stats.clear()
                for r in reversed(rollouts):
                    rollout_steps_iteration += r.total_steps

                    dataset = r.dataset(self.device)
                    dataloader = RolloutDataLoader(
                        dataset, batch_size=self.batch_size, shuffle=shuffle_minibatches
                    )
                    dataloader = accelerator.prepare(dataloader)

                    for (
                        mb_obs,
                        mb_actions,
                        mb_action_masks,
                        mb_logprobs,
                        mb_values,
                        mb_adv,
                        mb_returns,
                        mb_teacher_logprobs,
                    ) in dataloader:
                        policy.reset_noise(self.batch_size)

                        if self.normalize_advantages_after_scaling:
                            if multi_reward_weights is not None:
                                mb_adv = mb_adv @ multi_reward_weights

                            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                        else:
                            if self.normalize_advantage:
                                mb_adv = (mb_adv - mb_adv.mean(0)) / (
                                    mb_adv.std(0) + 1e-8
                                )
                            elif self.standardize_advantage:
                                mb_adv = mb_adv / (mb_adv.std(0) + 1e-8)
                            if multi_reward_weights is not None:
                                mb_adv = mb_adv @ multi_reward_weights

                        additional_losses = {}

                        with accelerator.accumulate(policy):
                            optimizer.zero_grad()
                            new_logprobs, entropy, new_values = policy(
                                mb_obs, mb_actions, action_masks=mb_action_masks
                            )

                            logratio = new_logprobs - mb_logprobs
                            ratio = torch.exp(logratio)
                            clipped_ratio = torch.clamp(
                                ratio, min=1 - pi_clip, max=1 + pi_clip
                            )
                            pi_loss = -torch.min(
                                ratio * mb_adv, clipped_ratio * mb_adv
                            ).mean()

                            v_loss_unclipped = self.vf_loss_fn(
                                new_values, mb_returns, reduction="none"
                            )
                            if v_clip is not None:
                                v_loss_clipped = self.vf_loss_fn(
                                    mb_values
                                    + torch.clamp(
                                        new_values - mb_values, -v_clip, v_clip
                                    ),
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

                            entropy_loss = -entropy.mean()
                            with torch.no_grad():
                                approx_kl = (
                                    ((ratio - 1) - logratio).mean().cpu().numpy().item()
                                )
                            if (
                                self.kl_cutoff is not None
                                and approx_kl > self.kl_cutoff
                            ):
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
                                ), "No teacher logprobs"
                                teacher_kl_loss = self.teacher_kl_loss_fn(
                                    new_logprobs,
                                    mb_teacher_logprobs,
                                    ratio
                                    if self.teacher_loss_importance_sampling
                                    else None,
                                )
                                additional_losses[
                                    "teacher_kl_loss"
                                ] = teacher_kl_loss.item()
                                loss += self.teacher_kl_loss_coef * teacher_kl_loss

                            loss /= minibatches_per_step

                            accelerator.backward(loss)
                            grad_norm = (
                                accelerator.clip_grad_norm_(
                                    policy.parameters(), self.max_grad_norm
                                ).item()
                                if accelerator.sync_gradients
                                else None
                            )
                            optimizer.step()

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
                            accelerator.gather_for_metrics(
                                DPPOTrainStepStats(
                                    loss.item(),
                                    pi_loss.item(),
                                    v_loss.detach().float().cpu().numpy(),
                                    entropy_loss.item(),
                                    approx_kl,
                                    clipped_frac,
                                    val_clipped_frac,
                                    additional_losses,
                                    grad_norm,
                                )
                            )
                        )

                    if rollout_iteration_cnt == 1:
                        y_true_list.append(r.y_true)
                        y_pred_list.append(r.y_pred)
                        timesteps_elapsed += r.total_steps
                    self.policy = accelerator.unwrap_model(policy)
                    learner_data_store_view.submit_learner_update(
                        LearnerDataStoreViewUpdate(self.policy, self, timesteps_elapsed)
                    )
                    n_epochs = (
                        rollout_iteration_cnt
                        - 1
                        + rollout_steps_iteration / total_rollout_steps
                    )
                    (next_rollouts,) = learner_data_store_view.get_learner_view(
                        wait=(
                            self.max_n_epochs is not None
                            and n_epochs >= self.max_n_epochs
                        )
                    )
                    if next_rollouts:
                        break
                if rollout_iteration_cnt == 1:
                    rollout_steps_elapsed = rollout_steps_iteration

            accelerator._dataloaders.clear()
            if (
                self.freeze_policy_head
                or self.freeze_value_head
                or self.freeze_backbone
            ):
                policy.unfreeze()

            self.tb_writer.on_timesteps_elapsed(timesteps_elapsed)

            y_true = np.concatenate(y_true_list)
            y_pred = np.concatenate(y_pred_list)
            var_y = np.var(y_true).item()
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred).item() / var_y
            )

            train_stats = DPPOTrainStats(
                step_stats,
                explained_var,
                n_epochs,
            )
            train_stats.write_to_tensorboard(self.tb_writer)

            end_time = perf_counter()
            self.tb_writer.add_scalar(
                "train/steps_per_second",
                rollout_steps_elapsed / (end_time - start_time),
            )

            if callbacks:
                if not all(
                    c.on_step(
                        timesteps_elapsed=rollout_steps_elapsed, train_stats=train_stats
                    )
                    for c in callbacks
                ):
                    logging.info(
                        f"Callback terminated training at {timesteps_elapsed} timesteps"
                    )
                    break
            rollouts = next_rollouts
            gc.collect()
        return self
