from typing import Optional, Type

import torch
import torch.nn as nn

from rl_algo_impls.shared.encoder.cnn import CnnFeatureExtractor
from rl_algo_impls.shared.module.module import layer_init


class NatureCnn(CnnFeatureExtractor):
    """
    CNN from DQN Nature paper: Mnih, Volodymyr, et al.
    "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.
    """

    def __init__(
        self,
        in_channels: int,
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: Optional[bool] = None,
        **kwargs,
    ) -> None:
        if init_layers_orthogonal is None:
            init_layers_orthogonal = True
        super().__init__(in_channels, activation, init_layers_orthogonal)
        self.cnn = nn.Sequential(
            layer_init(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                init_layers_orthogonal,
            ),
            activation(),
            layer_init(
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                init_layers_orthogonal,
            ),
            activation(),
            layer_init(
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                init_layers_orthogonal,
            ),
            activation(),
            nn.Flatten(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.cnn(obs)
