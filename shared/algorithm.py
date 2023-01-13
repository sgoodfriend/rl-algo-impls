import gym
import torch

from abc import ABC, abstractmethod
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from typing import List, Optional, Union

from shared.callbacks.callback import Callback
from shared.policy import Policy
from shared.stats import EpisodesStats


class Algorithm(ABC):

    @abstractmethod
    def __init__(self, policy: Policy, env: VecEnv, device: torch.device,
                 **kwargs) -> None:
        super().__init__()
        self.policy = policy
        self.env = env
        self.device = device

    @abstractmethod
    def learn(self,
              total_timesteps: int,
              callback: Optional[Callback] = None) -> List[EpisodesStats]:
        ...