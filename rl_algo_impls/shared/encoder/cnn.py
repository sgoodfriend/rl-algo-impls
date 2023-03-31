from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
import torch.nn as nn

from rl_algo_impls.shared.module.utils import layer_init

EncoderOutDim = Union[int, Tuple[int, ...]]


class CnnEncoder(nn.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        obs_space: gym.Space,
        **kwargs,
    ) -> None:
        super().__init__()
        self.range_size = np.max(obs_space.high) - np.min(obs_space.low)  # type: ignore

    def preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        return obs.float() / self.range_size

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.preprocess(obs)

    @property
    @abstractmethod
    def out_dim(self) -> EncoderOutDim:
        ...


class FlattenedCnnEncoder(CnnEncoder):
    def __init__(
        self,
        obs_space: gym.Space,
        activation: Type[nn.Module],
        linear_init_layers_orthogonal: bool,
        cnn_flatten_dim: int,
        cnn: nn.Module,
        **kwargs,
    ) -> None:
        super().__init__(obs_space, **kwargs)
        self.cnn = cnn
        self.flattened_dim = cnn_flatten_dim
        with torch.no_grad():
            cnn_out = torch.flatten(
                cnn(self.preprocess(torch.as_tensor(obs_space.sample()))), start_dim=1
            )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer_init(
                nn.Linear(cnn_out.shape[1], cnn_flatten_dim),
                linear_init_layers_orthogonal,
            ),
            activation(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = super().forward(obs)
        x = self.cnn(x)
        x = self.fc(x)
        return x

    @property
    def out_dim(self) -> EncoderOutDim:
        return self.flattened_dim
