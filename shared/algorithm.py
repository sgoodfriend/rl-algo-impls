import gym
import torch

from abc import ABC, abstractmethod
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from torch.utils.tensorboard.writer import SummaryWriter
from typing import List, Optional, TypeVar

from shared.callbacks.callback import Callback
from shared.policy.policy import Policy
from shared.stats import EpisodesStats

AlgorithmSelf = TypeVar("AlgorithmSelf", bound="Algorithm")

class Algorithm(ABC):
    @abstractmethod
    def __init__(
        self,
        policy: Policy,
        env: VecEnv,
        device: torch.device,
        tb_writer: SummaryWriter,
        **kwargs,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.env = env
        self.device = device
        self.tb_writer = tb_writer

    @abstractmethod
    def learn(
        self: AlgorithmSelf, total_timesteps: int, callback: Optional[Callback] = None
    ) -> AlgorithmSelf:
        ...
