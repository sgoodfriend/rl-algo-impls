import gym
import torch
import torch.nn as nn

from typing import Sequence, Type
from shared.module import FeatureExtractor, mlp


class CriticHead(nn.Module):
    def __init__(
        self,
        hidden_sizes: Sequence[int] = (32,),
        activation: Type[nn.Module] = nn.Tanh,
        init_layers_orthogonal: bool = True,
    ) -> None:
        super().__init__()
        layer_sizes = tuple(hidden_sizes) + (1,)
        self._fc = mlp(
            layer_sizes,
            activation,
            init_layers_orthogonal=init_layers_orthogonal,
            final_layer_gain=1.0,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        v = self._fc(obs)
        return v.squeeze(-1)


class HeadedCritic(nn.Module):
    def __init__(
        self,
        head: CriticHead,
        obs_space: gym.Space,
        hidden_sizes: Sequence[int] = (32,),
        activation: Type[nn.Module] = nn.Tanh,
        init_layers_orthogonal: bool = True,
    ) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            obs_space,
            activation,
            hidden_sizes[0],
            init_layers_orthogonal=init_layers_orthogonal,
        )
        self.head = head

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        fe = self.feature_extractor(obs)
        return self.head(fe)
