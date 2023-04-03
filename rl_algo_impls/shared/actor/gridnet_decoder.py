from typing import Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from rl_algo_impls.shared.actor import Actor, PiForward, pi_forward
from rl_algo_impls.shared.actor.gridnet import GridnetDistribution
from rl_algo_impls.shared.encoder import EncoderOutDim
from rl_algo_impls.shared.module.utils import layer_init


class Transpose(nn.Module):
    def __init__(self, permutation: Tuple[int, ...]) -> None:
        super().__init__()
        self.permutation = permutation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.permutation)


class GridnetDecoder(Actor):
    def __init__(
        self,
        map_size: int,
        action_vec: NDArray[np.int64],
        in_dim: EncoderOutDim,
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: bool = True,
    ) -> None:
        super().__init__()
        self.map_size = map_size
        self.action_vec = action_vec
        assert isinstance(in_dim, tuple)
        self.deconv = nn.Sequential(
            layer_init(
                nn.ConvTranspose2d(
                    in_dim[0], 128, 3, stride=2, padding=1, output_padding=1
                ),
                init_layers_orthogonal=init_layers_orthogonal,
            ),
            activation(),
            layer_init(
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                init_layers_orthogonal=init_layers_orthogonal,
            ),
            activation(),
            layer_init(
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                init_layers_orthogonal=init_layers_orthogonal,
            ),
            activation(),
            layer_init(
                nn.ConvTranspose2d(
                    32, action_vec.sum(), 3, stride=2, padding=1, output_padding=1
                ),
                init_layers_orthogonal=init_layers_orthogonal,
                std=0.01,
            ),
            Transpose((0, 2, 3, 1)),
        )

    def forward(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
    ) -> PiForward:
        assert (
            action_masks is not None
        ), f"No mask case unhandled in {self.__class__.__name__}"
        logits = self.deconv(obs)
        pi = GridnetDistribution(self.map_size, self.action_vec, logits, action_masks)
        return pi_forward(pi, actions)

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return (self.map_size, len(self.action_vec))
