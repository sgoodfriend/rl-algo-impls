import gym
import torch as th
import torch.nn as nn

from gym.spaces import Discrete
from typing import Sequence, Type

from shared.module import feature_extractor, mlp


class QNetwork(nn.Module):

    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            hidden_sizes: Sequence[int],
            activation: Type[nn.Module] = nn.ReLU,  # Used by stable-baselines3
    ) -> None:
        super().__init__()
        assert isinstance(action_space, Discrete)
        layer_sizes = tuple(hidden_sizes) + (action_space.n, )
        self._preprocessor, self._feature_extractor = feature_extractor(
            observation_space, activation, layer_sizes[0])
        self._fc = mlp(layer_sizes, activation)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = self._preprocessor(obs) if self._preprocessor else obs
        x = self._feature_extractor(x)
        return self._fc(x)
