from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar

import gym
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.shared.callbacks import Callback
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv

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
        self: AlgorithmSelf,
        train_timesteps: int,
        callbacks: Optional[List[Callback]] = None,
        total_timesteps: Optional[int] = None,
        start_timesteps: int = 0,
    ) -> AlgorithmSelf:
        ...
