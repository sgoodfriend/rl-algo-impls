import gym
import numpy as np
import torch

from abc import ABC, abstractmethod
from gym.spaces import Box, Discrete
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs
from typing import TypeVar, Union

PolicySelf = TypeVar("PolicySelf", bound="Policy")


class Policy(ABC):

    @abstractmethod
    def __init__(self, env: VecEnv, device: torch.device, **kwargs) -> None:
        super().__init__()
        self.env = env
        self.device = device
        self.training = True

    @abstractmethod
    def act(self, obs: VecEnvObs) -> np.ndarray:
        ...

    def train(self: PolicySelf, mode: bool = True) -> PolicySelf:
        self.training = mode
        return self

    def eval(self: PolicySelf) -> PolicySelf:
        return self.train(False)

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...