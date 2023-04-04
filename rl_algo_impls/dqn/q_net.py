from typing import Optional, Sequence, Type

import gym
import torch as th
import torch.nn as nn
from gym.spaces import Discrete

from rl_algo_impls.shared.encoder import Encoder
from rl_algo_impls.shared.module.utils import mlp


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_sizes: Sequence[int] = [],
        activation: Type[nn.Module] = nn.ReLU,  # Used by stable-baselines3
        cnn_flatten_dim: int = 512,
        cnn_style: str = "nature",
        cnn_layers_init_orthogonal: Optional[bool] = None,
        impala_channels: Sequence[int] = (16, 32, 32),
    ) -> None:
        super().__init__()
        assert isinstance(action_space, Discrete)
        self._feature_extractor = Encoder(
            observation_space,
            activation,
            cnn_flatten_dim=cnn_flatten_dim,
            cnn_style=cnn_style,
            cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
            impala_channels=impala_channels,
        )
        layer_sizes = (
            (self._feature_extractor.out_dim,) + tuple(hidden_sizes) + (action_space.n,)
        )
        self._fc = mlp(layer_sizes, activation)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = self._feature_extractor(obs)
        return self._fc(x)
