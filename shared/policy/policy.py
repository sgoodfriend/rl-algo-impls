import numpy as np
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs
from typing import Dict, Optional, Type, TypeVar, Union

PolicySelf = TypeVar("PolicySelf", bound="Policy")

ACTIVATION: Dict[str, Type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}


class Policy(nn.Module, ABC):
    @abstractmethod
    def __init__(self, env: VecEnv, **kwargs) -> None:
        super().__init__()
        self.env = env
        self.device = None

    def to(
        self: PolicySelf,
        device: Optional[torch.device] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
        non_blocking: bool = False,
    ) -> PolicySelf:
        super().to(device, dtype, non_blocking)
        self.device = device
        return self

    @abstractmethod
    def act(self, obs: VecEnvObs) -> np.ndarray:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...
