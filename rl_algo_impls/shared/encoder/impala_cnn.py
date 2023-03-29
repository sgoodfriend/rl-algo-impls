from typing import Optional, Sequence, Type

import torch
import torch.nn as nn

from rl_algo_impls.shared.encoder.cnn import CnnFeatureExtractor
from rl_algo_impls.shared.module.module import layer_init


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: bool = False,
    ) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            activation(),
            layer_init(
                nn.Conv2d(channels, channels, 3, padding=1), init_layers_orthogonal
            ),
            activation(),
            layer_init(
                nn.Conv2d(channels, channels, 3, padding=1), init_layers_orthogonal
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)


class ConvSequence(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: bool = False,
    ) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            layer_init(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                init_layers_orthogonal,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResidualBlock(out_channels, activation, init_layers_orthogonal),
            ResidualBlock(out_channels, activation, init_layers_orthogonal),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ImpalaCnn(CnnFeatureExtractor):
    """
    IMPALA-style CNN architecture
    """

    def __init__(
        self,
        in_channels: int,
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: Optional[bool] = None,
        impala_channels: Sequence[int] = (16, 32, 32),
        **kwargs,
    ) -> None:
        if init_layers_orthogonal is None:
            init_layers_orthogonal = False
        super().__init__(in_channels, activation, init_layers_orthogonal)
        sequences = []
        for out_channels in impala_channels:
            sequences.append(
                ConvSequence(
                    in_channels, out_channels, activation, init_layers_orthogonal
                )
            )
            in_channels = out_channels
        sequences.extend(
            [
                activation(),
                nn.Flatten(),
            ]
        )
        self.seq = nn.Sequential(*sequences)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.seq(obs)
