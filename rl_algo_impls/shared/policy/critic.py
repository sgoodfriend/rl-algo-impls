from typing import Sequence, Type

import numpy as np
import torch
import torch.nn as nn

from rl_algo_impls.shared.encoder import EncoderOutDim
from rl_algo_impls.shared.module.utils import mlp


class CriticHead(nn.Module):
    def __init__(
        self,
        in_dim: EncoderOutDim,
        hidden_sizes: Sequence[int] = (),
        activation: Type[nn.Module] = nn.Tanh,
        init_layers_orthogonal: bool = True,
    ) -> None:
        super().__init__()
        seq = []
        if isinstance(in_dim, tuple):
            seq.append(nn.Flatten())
            in_channels = int(np.prod(in_dim))
        else:
            in_channels = in_dim
        layer_sizes = (in_channels,) + tuple(hidden_sizes) + (1,)
        seq.append(
            mlp(
                layer_sizes,
                activation,
                init_layers_orthogonal=init_layers_orthogonal,
                final_layer_gain=1.0,
                hidden_layer_gain=1.0,
            )
        )
        self._fc = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        v = self._fc(obs)
        return v.squeeze(-1)
