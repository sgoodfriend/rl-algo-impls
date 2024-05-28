import logging
from time import perf_counter
from typing import Dict, List, Optional, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from rl_algo_impls.acbc.train_stats import TrainStats
from rl_algo_impls.rollout.acbc_rollout import ACBCBatch
from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.callbacks.callback import Callback
from rl_algo_impls.shared.data_store.data_store_data import LearnerDataStoreViewUpdate
from rl_algo_impls.shared.data_store.data_store_view import LearnerDataStoreView
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.schedule import update_learning_rate
from rl_algo_impls.shared.stats import log_scalars
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.shared.tensor_utils import NumOrList, num_or_array

"""
Actor-Critic Behavior Cloning with Critic Bootstrapping
"""

ACBCSelf = TypeVar("ACBCSelf", bound="ACBC")


class ACBC(Algorithm):
    def __init__(
        self,
        policy: ActorCritic,
        device: torch.device,
        tb_writer: AbstractSummaryWrapper,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        n_epochs: int = 10,
        vf_coef: NumOrList = 0.25,
        max_grad_norm: float = 0.5,
        gradient_accumulation: bool = False,
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

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.vf_coef = num_or_array(vf_coef)
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation = gradient_accumulation
        self.scale_loss_by_num_actions = scale_loss_by_num_actions

    def learn(
        self: ACBCSelf,
        learner_data_store_view: LearnerDataStoreView,
        train_timesteps: int,
        callbacks: Optional[List[Callback]] = None,
    ) -> ACBCSelf:
        timesteps_elapsed = 0
        while timesteps_elapsed < train_timesteps:
            start_time = perf_counter()

            update_learning_rate(self.optimizer, self.learning_rate)

            chart_scalars = {
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "vf_coef": self.vf_coef,
            }
            log_scalars(self.tb_writer, "charts", chart_scalars)

            (rollouts,) = learner_data_store_view.get_learner_view()
            if len(rollouts) > 1:
                logging.warning(
                    f"A2C does not support multiple rollouts ({len(rollouts)}) per epoch. "
                    "Only the last rollout will be used"
                )
            r = rollouts[-1]
            timesteps_elapsed += r.total_steps

            step_stats: List[Dict[str, Union[float, np.ndarray]]] = []
            vf_coef = torch.Tensor(np.array(self.vf_coef)).to(self.device)
            for _ in range(self.n_epochs):
                step_stats.clear()
                for mb in r.minibatches(
                    self.batch_size, self.device, shuffle=not self.gradient_accumulation
                ):
                    self.policy.reset_noise(self.batch_size)
                    assert isinstance(mb, ACBCBatch)
                    (
                        mb_obs,
                        mb_actions,
                        mb_action_masks,
                        mb_returns,
                        mb_num_actions,
                    ) = mb

                    new_logprobs, _, new_values = self.policy(
                        mb_obs, mb_actions, action_masks=mb_action_masks
                    )
                    if self.scale_loss_by_num_actions:
                        pi_loss = -torch.where(
                            mb_num_actions > 0, new_logprobs / mb_num_actions, 0
                        ).mean()
                    else:
                        pi_loss = -new_logprobs.mean()
                    v_loss = ((new_values - mb_returns) ** 2).mean(0)
                    loss = pi_loss + (vf_coef * v_loss).sum()

                    if self.gradient_accumulation:
                        loss /= r.num_minibatches(self.batch_size)
                    loss.backward()
                    if not self.gradient_accumulation:
                        self.optimizer_step()

                    step_stats.append(
                        {
                            "loss": loss.item(),
                            "pi_loss": pi_loss.item(),
                            "v_loss": v_loss.detach().cpu().numpy(),
                        }
                    )

                if self.gradient_accumulation:
                    self.optimizer_step()

            self.tb_writer.on_timesteps_elapsed(timesteps_elapsed)

            var_y = np.var(r.y_true).item()
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(r.y_true - r.y_pred).item() / var_y
            )
            TrainStats(step_stats, explained_var).write_to_tensorboard(self.tb_writer)

            end_time = perf_counter()
            rollout_steps = r.total_steps
            self.tb_writer.add_scalar(
                "train/steps_per_second",
                rollout_steps / (end_time - start_time),
            )

            if callbacks:
                if not all(
                    c.on_step(timesteps_elapsed=rollout_steps) for c in callbacks
                ):
                    logging.info(
                        f"Callback terminated training at {timesteps_elapsed} timesteps"
                    )
                    learner_data_store_view.submit_learner_update(
                        LearnerDataStoreViewUpdate(self.policy, self, timesteps_elapsed)
                    )
                    break

            learner_data_store_view.submit_learner_update(
                LearnerDataStoreViewUpdate(self.policy, self, timesteps_elapsed)
            )
        return self

    def optimizer_step(self) -> None:
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
