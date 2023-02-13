import numpy as np
import os
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from stable_baselines3.common.vec_env import unwrap_vec_normalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs
from typing import Dict, Optional, Type, TypeVar, Union

ACTIVATION: Dict[str, Type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}

VEC_NORMALIZE_FILENAME = "vecnormalize.pkl"
MODEL_FILENAME = "model.pth"

PolicySelf = TypeVar("PolicySelf", bound="Policy")


class Policy(nn.Module, ABC):
    @abstractmethod
    def __init__(self, env: VecEnv, **kwargs) -> None:
        super().__init__()
        self.env = env
        self.vec_normalize = unwrap_vec_normalize(env)
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
    def act(self, obs: VecEnvObs, deterministic: bool = True) -> np.ndarray:
        ...

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        if self.vec_normalize:
            self.vec_normalize.save(os.path.join(path, VEC_NORMALIZE_FILENAME))
        torch.save(
            self.state_dict(),
            os.path.join(path, MODEL_FILENAME),
        )

    def load(self, path: str) -> None:
        # VecNormalize load occurs in env.py
        self.load_state_dict(
            torch.load(os.path.join(path, MODEL_FILENAME), map_location=self.device)
        )

    def reset_noise(self) -> None:
        pass

    def _as_tensor(self, obs: VecEnvObs) -> torch.Tensor:
        assert isinstance(obs, np.ndarray)
        o = torch.as_tensor(obs)
        if self.device is not None:
            o = o.to(self.device)
        return o