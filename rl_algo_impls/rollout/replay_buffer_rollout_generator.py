import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch

from rl_algo_impls.rollout.rollout import RolloutGenerator
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.wrappers.vector_wrapper import VectorEnv


class Batch(NamedTuple):
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_obs: torch.Tensor


class Transition(NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    next_obs: np.ndarray


class ReplayBufferRolloutGenerator(RolloutGenerator):
    def __init__(
        self,
        policy: Policy,
        vec_env: VectorEnv,
        buffer_size: int = 1_000_000,
        learning_starts: int = 50_000,
        train_freq: int = 4,
    ) -> None:
        super().__init__(policy, vec_env)
        self.policy = policy
        self.vec_env = vec_env
        self.next_obs, _ = vec_env.reset()

        self.learning_starts = learning_starts
        self.train_freq = train_freq

        self.buffer = deque(maxlen=buffer_size)
        if self.learning_starts > 0:
            self._collect_transitions(
                int(np.ceil(self.learning_starts / self.vec_env.num_envs)), eps=1
            )

    def rollout(self, **kwargs) -> int:
        return self._collect_transitions(self.train_freq, **kwargs)

    def _collect_transitions(self, n_steps: int, **kwargs) -> int:
        self.policy.train(False)
        for _ in range(n_steps):
            actions = self.policy.act(self.next_obs, deterministic=False, **kwargs)
            next_obs, rewards, terminations, truncations, _ = self.vec_env.step(actions)
            dones = terminations | truncations
            assert isinstance(self.next_obs, np.ndarray)
            assert isinstance(next_obs, np.ndarray)
            for i in range(self.vec_env.num_envs):
                self.buffer.append(
                    Transition(
                        self.next_obs[i], actions[i], rewards[i], dones[i], next_obs[i]
                    )
                )
            self.next_obs = next_obs
        return n_steps * self.vec_env.num_envs

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        ts = random.sample(self.buffer, batch_size)
        return Batch(
            obs=torch.as_tensor([t.obs for t in ts], device=device),
            actions=torch.as_tensor([t.action for t in ts], device=device),
            rewards=torch.as_tensor(
                [t.reward for t in ts], dtype=torch.float32, device=device
            ),
            dones=torch.as_tensor(
                [t.done for t in ts], dtype=torch.bool, device=device
            ),
            next_obs=torch.as_tensor([t.next_obs for t in ts], device=device),
        )
