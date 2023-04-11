import copy
import logging
import random
from collections import deque
from typing import List, NamedTuple, Optional, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.dqn.policy import DQNPolicy
from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.callbacks import Callback
from rl_algo_impls.shared.schedule import linear_schedule
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv, VecEnvObs


class Transition(NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    next_obs: np.ndarray


class Batch(NamedTuple):
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    next_obs: np.ndarray


class ReplayBuffer:
    def __init__(self, num_envs: int, maxlen: int) -> None:
        self.num_envs = num_envs
        self.buffer = deque(maxlen=maxlen)

    def add(
        self,
        obs: VecEnvObs,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_obs: VecEnvObs,
    ) -> None:
        assert isinstance(obs, np.ndarray)
        assert isinstance(next_obs, np.ndarray)
        for i in range(self.num_envs):
            self.buffer.append(
                Transition(obs[i], action[i], reward[i], done[i], next_obs[i])
            )

    def sample(self, batch_size: int) -> Batch:
        ts = random.sample(self.buffer, batch_size)
        return Batch(
            obs=np.array([t.obs for t in ts]),
            actions=np.array([t.action for t in ts]),
            rewards=np.array([t.reward for t in ts]),
            dones=np.array([t.done for t in ts]),
            next_obs=np.array([t.next_obs for t in ts]),
        )

    def __len__(self) -> int:
        return len(self.buffer)


DQNSelf = TypeVar("DQNSelf", bound="DQN")


class DQN(Algorithm):
    def __init__(
        self,
        policy: DQNPolicy,
        env: VecEnv,
        device: torch.device,
        tb_writer: SummaryWriter,
        learning_rate: float = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 50_000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 10_000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10.0,
    ) -> None:
        super().__init__(policy, env, device, tb_writer)
        self.policy = policy

        self.optimizer = Adam(self.policy.q_net.parameters(), lr=learning_rate)

        self.target_q_net = copy.deepcopy(self.policy.q_net).to(self.device)
        self.target_q_net.train(False)
        self.tau = tau
        self.target_update_interval = target_update_interval

        self.replay_buffer = ReplayBuffer(self.env.num_envs, buffer_size)
        self.batch_size = batch_size

        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps

        self.gamma = gamma
        self.exploration_eps_schedule = linear_schedule(
            exploration_initial_eps,
            exploration_final_eps,
            end_fraction=exploration_fraction,
        )

        self.max_grad_norm = max_grad_norm

    def learn(
        self: DQNSelf, total_timesteps: int, callbacks: Optional[List[Callback]] = None
    ) -> DQNSelf:
        self.policy.train(True)
        obs = self.env.reset()
        obs = self._collect_rollout(self.learning_starts, obs, 1)
        learning_steps = total_timesteps - self.learning_starts
        timesteps_elapsed = 0
        steps_since_target_update = 0
        while timesteps_elapsed < learning_steps:
            progress = timesteps_elapsed / learning_steps
            eps = self.exploration_eps_schedule(progress)
            obs = self._collect_rollout(self.train_freq, obs, eps)
            rollout_steps = self.train_freq
            timesteps_elapsed += rollout_steps
            for _ in range(
                self.gradient_steps if self.gradient_steps > 0 else self.train_freq
            ):
                self.train()
            steps_since_target_update += rollout_steps
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

    def train(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return
        o, a, r, d, next_o = self.replay_buffer.sample(self.batch_size)
        o = torch.as_tensor(o, device=self.device)
        a = torch.as_tensor(a, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.long, device=self.device)
        next_o = torch.as_tensor(next_o, device=self.device)

        with torch.no_grad():
            target = r + (1 - d) * self.gamma * self.target_q_net(next_o).max(1).values
        current = self.policy.q_net(o).gather(dim=1, index=a).squeeze(1)
        loss = F.smooth_l1_loss(current, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.policy.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def _collect_rollout(self, timesteps: int, obs: VecEnvObs, eps: float) -> VecEnvObs:
        for _ in range(0, timesteps, self.env.num_envs):
            action = self.policy.act(obs, eps, deterministic=False)
            next_obs, reward, done, _ = self.env.step(action)
            self.replay_buffer.add(obs, action, reward, done, next_obs)
            obs = next_obs
        return obs

    def _update_target(self) -> None:
        for target_param, param in zip(
            self.target_q_net.parameters(), self.policy.q_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
