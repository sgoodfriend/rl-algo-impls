from typing import Optional, Tuple, Type, Union

import gym
import torch
import torch.nn as nn

from rl_algo_impls.shared.encoder.cnn import CnnEncoder, EncoderOutDim
from rl_algo_impls.shared.module.utils import layer_init


class GridnetEncoder(CnnEncoder):
    """
    Encoder for encoder-decoder for Gym-MicroRTS
    """

    def __init__(
        self,
        obs_space: gym.Space,
        activation: Type[nn.Module] = nn.ReLU,
        cnn_init_layers_orthogonal: Optional[bool] = None,
        **kwargs
    ) -> None:
        if cnn_init_layers_orthogonal is None:
            cnn_init_layers_orthogonal = True
        super().__init__(obs_space, **kwargs)
        in_channels = obs_space.shape[0]  # type: ignore
        self.encoder = nn.Sequential(
            layer_init(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                cnn_init_layers_orthogonal,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
            activation(),
            layer_init(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                cnn_init_layers_orthogonal,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
            activation(),
            layer_init(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                cnn_init_layers_orthogonal,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
            activation(),
            layer_init(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                cnn_init_layers_orthogonal,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
            activation(),
        )
        with torch.no_grad():
            encoder_out = self.encoder(
                self.preprocess(torch.as_tensor(obs_space.sample()))  # type: ignore
            )
            self._out_dim = encoder_out.shape[1:]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(super().forward(obs))

    @property
    def out_dim(self) -> EncoderOutDim:
        return self._out_dim
