from typing import Optional, Type

import gym
import torch
import torch.nn as nn

from rl_algo_impls.shared.encoder.cnn import FlattenedCnnEncoder
from rl_algo_impls.shared.module.utils import layer_init


class MicrortsCnn(FlattenedCnnEncoder):
    """
    Base CNN architecture for Gym-MicroRTS
    """

    def __init__(
        self,
        obs_space: gym.Space,
        activation: Type[nn.Module],
        cnn_init_layers_orthogonal: Optional[bool],
        linear_init_layers_orthogonal: bool,
        cnn_flatten_dim: int,
        **kwargs,
    ) -> None:
        if cnn_init_layers_orthogonal is None:
            cnn_init_layers_orthogonal = True
        in_channels = obs_space.shape[0]  # type: ignore
        cnn = nn.Sequential(
            layer_init(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=2),
                cnn_init_layers_orthogonal,
            ),
            activation(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2), cnn_init_layers_orthogonal),
            activation(),
            nn.Flatten(),
        )
        super().__init__(
            obs_space,
            activation,
            linear_init_layers_orthogonal,
            cnn_flatten_dim,
            cnn,
            **kwargs,
        )
