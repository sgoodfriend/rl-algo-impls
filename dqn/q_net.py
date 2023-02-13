import gym
import torch as th
import torch.nn as nn

from gym.spaces import Discrete
from typing import Sequence, Type

from shared.feature_extractor import FeatureExtractor
from shared.module import mlp


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_sizes: Sequence[int] = [],
        activation: Type[nn.Module] = nn.ReLU,  # Used by stable-baselines3
    ) -> None:
        super().__init__()
        assert isinstance(action_space, Discrete)
        self._feature_extractor = FeatureExtractor(observation_space, activation)
        layer_sizes = (
            (self._feature_extractor.out_dim,) + tuple(hidden_sizes) + (action_space.n,)
        )
        self._fc = mlp(layer_sizes, activation)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = self._feature_extractor(obs)
        return self._fc(x)
