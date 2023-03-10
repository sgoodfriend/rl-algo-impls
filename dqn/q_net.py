import gym
import torch as th
import torch.nn as nn

from gym.spaces import Discrete
from typing import Optional, Sequence, Type

from shared.module.feature_extractor import FeatureExtractor
from shared.module.module import mlp


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_sizes: Sequence[int] = [],
        activation: Type[nn.Module] = nn.ReLU,  # Used by stable-baselines3
        cnn_feature_dim: int = 512,
        cnn_style: str = "nature",
        cnn_layers_init_orthogonal: Optional[bool] = None,
        impala_channels: Sequence[int] = (16, 32, 32),
    ) -> None:
        super().__init__()
        assert isinstance(action_space, Discrete)
        self._feature_extractor = FeatureExtractor(
            observation_space,
            activation,
            cnn_feature_dim=cnn_feature_dim,
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
