import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch

from rl_algo_impls.rollout.synchronous_rollout_generator import (
    SynchronousRolloutGenerator,
)
from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.stats import log_scalars
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)


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


class ReplayBufferRolloutGenerator(SynchronousRolloutGenerator):
    def __init__(
        self,
        config: Config,
        data_store_accessor: AbstractDataStoreAccessor,
        tb_writer: AbstractSummaryWrapper,
        buffer_size: int = 1_000_000,
        learning_starts: int = 50_000,
        train_freq: int = 4,
    ) -> None:
        super().__init__(
            config,
            data_store_accessor,
            tb_writer,
        )
        self.next_obs = np.zeros(
            (self.env_spaces.num_envs,)
            + self.env_spaces.single_observation_space.shape,
            dtype=self.env_spaces.single_observation_space.dtype,
        )

        self.learning_starts = learning_starts
        self.train_freq = train_freq

        self.buffer = deque(maxlen=buffer_size)
        if self.learning_starts > 0:
            self._collect_transitions(
                int(np.ceil(self.learning_starts / self.vec_env.num_envs)), eps=1
            )

    def prepare(self) -> None:
        self.next_obs, _ = self.vec_env.reset()

    def rollout(self, **kwargs) -> int:
        return self._collect_transitions(self.train_freq, **kwargs)

    def _collect_transitions(self, n_steps: int, **kwargs) -> int:
        rollout_view = self.data_store_view.update_for_rollout_start()
        if rollout_view is None:
            return 0
        (
            policy,
            rollout_params,
            self.tb_writer.timesteps_elapsed,
        ) = rollout_view
        self.update_rollout_params(rollout_params)
        log_scalars(self.tb_writer, "charts", rollout_params)

        policy.train(False)
        for _ in range(n_steps):
            actions = policy.act(self.next_obs, deterministic=False, **kwargs)
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
