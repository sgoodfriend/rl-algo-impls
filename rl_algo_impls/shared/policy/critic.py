import gym
import torch
import torch.nn as nn

from typing import Sequence, Type

from rl_algo_impls.shared.module.module import mlp


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
