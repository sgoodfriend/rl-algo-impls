import logging
from dataclasses import astuple
from time import perf_counter
from typing import Dict, List, Optional, TypeVar

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.acbc.train_stats import TrainStats
from rl_algo_impls.ppo.rollout import RolloutGenerator
from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.callbacks.callback import Callback
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.schedule import schedule, update_learning_rate
from rl_algo_impls.shared.stats import log_scalars

"""
Actor-Critic Behavior Cloning with Critic Bootstrapping
"""

ACBCSelf = TypeVar("ACBCSelf", bound="ACBC")


class ACBC(Algorithm):
    def __init__(
        self,
        policy: ActorCritic,
        device: torch.device,
        tb_writer: SummaryWriter,
        learning_rate: float = 3e-4,
        learning_rate_decay: str = "none",
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        vf_coef: float = 0.25,
        max_grad_norm: float = 0.5,
        update_returns_between_epochs: bool = False,
        gradient_accumulation: bool = False,
    ) -> None:
        super().__init__(policy, device, tb_writer)
        self.policy = policy

        self.learning_rate_schedule = schedule(learning_rate_decay, learning_rate)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_returns_between_epochs = update_returns_between_epochs
        self.gradient_accumulation = gradient_accumulation

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

        optimizer = None

        while timesteps_elapsed < start_timesteps + train_timesteps:
            start_time = perf_counter()

            progress = timesteps_elapsed / total_timesteps
            learning_rate = self.learning_rate_schedule(progress)
            if not optimizer:
                optimizer = Adam(self.policy.parameters(), lr=learning_rate)
            else:
                update_learning_rate(optimizer, learning_rate)

            chart_scalars = {
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            log_scalars(self.tb_writer, "charts", chart_scalars, timesteps_elapsed)

            r = rollout_generator.rollout()
            timesteps_elapsed += r.total_steps

            step_stats: List[Dict[str, float]] = []
            for e in range(self.n_epochs):
                if e == 0 or self.update_returns_between_epochs:
                    # Using the same value head computation as PPO since the idea is
                    # that the BC model will be the starting point for PPO.
                    r.update_advantages(
                        self.policy,
                        self.gamma,
                        self.gae_lambda,
                        update_returns=self.update_returns_between_epochs,
                    )
                step_stats.clear()
                for mb in r.minibatches(self.batch_size, self.device):
                    self.policy.reset_noise(self.batch_size)

                    (
                        mb_obs,
                        mb_logprobs,
                        mb_actions,
                        mb_action_masks,
                        _,
                        _,
                        mb_returns,
                    ) = astuple(mb)

                    new_logprobs, _, new_values = self.policy(
                        mb_obs, mb_actions, action_masks=mb_action_masks
                    )

                    pi_loss = -new_logprobs.mean()
                    v_loss = ((new_values - mb_returns) ** 2).mean(0)
                    loss = pi_loss + self.vf_coef * v_loss

                    if self.gradient_accumulation:
                        loss /= r.num_minibatches(self.batch_size)
                    loss.backward()
                    if not self.gradient_accumulation:
                        self.optimizer_step(optimizer)

                    with torch.no_grad():
                        logratio = new_logprobs - mb_logprobs
                        ratio = torch.exp(logratio)
                        approx_kl = ((ratio - 1) - logratio).mean().cpu().numpy().item()
                        step_stats.append(
                            {
                                "loss": loss.item(),
                                "pi_loss": pi_loss.item(),
                                "v_loss": v_loss.item(),
                                "approx_kl": approx_kl,
                            }
                        )
                if self.gradient_accumulation:
                    self.optimizer_step(optimizer)

                var_y = np.var(r.y_true).item()
                explained_var = (
                    np.nan
                    if var_y == 0
                    else 1 - np.var(r.y_true - r.y_pred).item() / var_y
                )
                TrainStats(step_stats, explained_var).write_to_tensorboard(
                    self.tb_writer, timesteps_elapsed
                )

                end_time = perf_counter()
                rollout_steps = r.total_steps
                self.tb_writer.add_scalar(
                    "train/steps_per_second",
                    rollout_steps / (end_time - start_time),
                    timesteps_elapsed,
                )

                if callbacks:
                    if not all(
                        c.on_step(timesteps_elapsed=rollout_steps) for c in callbacks
                    ):
                        logging.info(
                            f"Callback terminated training at {timesteps_elapsed} timesteps"
                        )
                        break
        return self

    def optimizer_step(self, optimizer: Optimizer) -> None:
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
