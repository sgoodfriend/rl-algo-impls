import copy
import logging
from typing import List, Optional, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.dqn.policy import DQNPolicy
from rl_algo_impls.rollout.replay_buffer_rollout_generator import (
    Batch,
    ReplayBufferRolloutGenerator,
)
from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.callbacks import Callback
from rl_algo_impls.shared.schedule import linear_schedule

DQNSelf = TypeVar("DQNSelf", bound="DQN")


class DQN(Algorithm):
    def __init__(
        self,
        policy: DQNPolicy,
        device: torch.device,
        tb_writer: SummaryWriter,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        gradient_steps: int = 1,
        target_update_interval: int = 10_000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10.0,
    ) -> None:
        super().__init__(
            policy,
            device,
            tb_writer,
            learning_rate,
            Adam(policy.q_net.parameters(), lr=learning_rate),
        )
        self.policy = policy

        self.target_q_net = copy.deepcopy(self.policy.q_net).to(self.device)
        self.target_q_net.train(False)
        self.tau = tau
        self.target_update_interval = target_update_interval

        self.batch_size = batch_size

        self.gradient_steps = gradient_steps

        self.gamma = gamma
        self.exploration_eps_schedule = linear_schedule(
            exploration_initial_eps,
            exploration_final_eps,
            end_progress=exploration_fraction,
        )

        self.max_grad_norm = max_grad_norm

    def learn(
        self: DQNSelf,
        train_timesteps: int,
        rollout_generator: ReplayBufferRolloutGenerator,
        callbacks: Optional[List[Callback]] = None,
        total_timesteps: Optional[int] = None,
        start_timesteps: int = 0,
    ) -> DQNSelf:
        if total_timesteps is None:
            total_timesteps = train_timesteps
        assert start_timesteps + train_timesteps <= total_timesteps

        timesteps_elapsed = max(start_timesteps, rollout_generator.learning_starts)
        steps_since_target_update = 0
        while timesteps_elapsed < total_timesteps:
            progress = timesteps_elapsed / total_timesteps
            eps = self.exploration_eps_schedule(progress)

            rollout_steps = rollout_generator.rollout(eps=eps)
            timesteps_elapsed += rollout_steps

            if len(rollout_generator.buffer) > self.batch_size:
                for _ in range(
                    self.gradient_steps
                    if self.gradient_steps > 0
                    else rollout_generator.train_freq
                ):
                    self.train(rollout_generator.sample(self.batch_size, self.device))
                steps_since_target_update += rollout_generator.train_freq
                if steps_since_target_update >= self.target_update_interval:
                    self._update_target()
                    steps_since_target_update = 0
            if callbacks:
                if not all(
                    c.on_step(timesteps_elapsed=rollout_steps) for c in callbacks
                ):
                    logging.info(
                        f"Callback terminated training at {timesteps_elapsed} timesteps"
                    )
                    break
        return self

    def train(self, batch: Batch) -> None:
        self.policy.train(True)
        o, a, r, d, next_o = batch

        with torch.no_grad():
            target = r + ~d * self.gamma * self.target_q_net(next_o).max(1).values
        current = self.policy.q_net(o).gather(dim=1, index=a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(current, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.policy.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def _update_target(self) -> None:
        for target_param, param in zip(
            self.target_q_net.parameters(), self.policy.q_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
