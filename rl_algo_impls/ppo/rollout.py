from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import numpy as np


class Rollout(NamedTuple):
    next_obs: np.ndarray
    next_action_masks: Optional[np.ndarray]
    next_episode_starts: np.ndarray

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    episode_starts: np.ndarray
    values: np.ndarray
    logprobs: np.ndarray
    action_masks: Optional[np.ndarray]

    total_steps: int


class RolloutGenerator(ABC):
    def __init__(self, n_steps: int, sde_sample_freq: int) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.sde_sample_freq = sde_sample_freq

    @abstractmethod
    def rollout(self) -> Rollout:
        ...
