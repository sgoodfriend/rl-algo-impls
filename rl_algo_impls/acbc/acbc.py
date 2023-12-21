import logging
from dataclasses import astuple
from time import perf_counter
from typing import Dict, List, Optional, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from rl_algo_impls.acbc.train_stats import TrainStats
from rl_algo_impls.ppo.ppo import NL
from rl_algo_impls.rollout.rollout import RolloutGenerator
from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.callbacks.callback import Callback
from rl_algo_impls.shared.callbacks.summary_wrapper import SummaryWrapper
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.schedule import update_learning_rate
from rl_algo_impls.shared.stats import log_scalars
from rl_algo_impls.shared.tensor_utils import num_or_array

"""
Actor-Critic Behavior Cloning with Critic Bootstrapping
"""

ACBCSelf = TypeVar("ACBCSelf", bound="ACBC")


class ACBC(Algorithm):
    def __init__(
        self,
        policy: ActorCritic,
        device: torch.device,
        tb_writer: SummaryWrapper,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: NL = 0.99,
        gae_lambda: NL = 0.95,
        vf_coef: NL = 0.25,
        max_grad_norm: float = 0.5,
        gradient_accumulation: bool = False,
        scale_loss_by_num_actions: bool = False,
    ) -> None:
        super().__init__(
            policy,
            device,
            tb_writer,
            learning_rate,
            Adam(policy.parameters(), lr=learning_rate),
        )
        self.policy = policy

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = num_or_array(gamma)
        self.gae_lambda = num_or_array(gae_lambda)
        self.vf_coef = num_or_array(vf_coef)
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation = gradient_accumulation
        self.scale_loss_by_num_actions = scale_loss_by_num_actions

    def learn(
        self: ACBCSelf,
        train_timesteps: int,
        rollout_generator: RolloutGenerator,
        callbacks: Optional[List[Callback]] = None,
        total_timesteps: Optional[int] = None,
        start_timesteps: int = 0,
    ) -> ACBCSelf:
        if total_timesteps is None:
            total_timesteps = train_timesteps
        assert start_timesteps + train_timesteps <= total_timesteps
        timesteps_elapsed = start_timesteps

        while timesteps_elapsed < start_timesteps + train_timesteps:
            start_time = perf_counter()

            update_learning_rate(self.optimizer, self.learning_rate)

            chart_scalars = {
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "vf_coef": self.vf_coef,
            }
            log_scalars(self.tb_writer, "charts", chart_scalars, timesteps_elapsed)

            r = rollout_generator.rollout(self.gamma, self.gae_lambda)
            timesteps_elapsed += r.total_steps

            step_stats: List[Dict[str, Union[float, np.ndarray]]] = []
            vf_coef = torch.Tensor(np.array(self.vf_coef)).to(self.device)
            for e in range(self.n_epochs):
                step_stats.clear()
                for mb in r.minibatches(
                    self.batch_size, shuffle=not self.gradient_accumulation
                ):
                    self.policy.reset_noise(self.batch_size)

                    (
                        mb_obs,
                        _,
                        mb_actions,
                        mb_action_masks,
                        mb_num_actions,
                        _,
                        _,
                        mb_returns,
                        _,  # mb_additional,
                    ) = astuple(mb)

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

            self.tb_writer.on_steps(rollout_steps)
            if callbacks:
                if not all(
                    c.on_step(timesteps_elapsed=rollout_steps) for c in callbacks
                ):
                    logging.info(
                        f"Callback terminated training at {timesteps_elapsed} timesteps"
                    )
                    break
        return self

    def optimizer_step(self) -> None:
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
