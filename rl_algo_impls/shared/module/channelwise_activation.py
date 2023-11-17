from typing import Sequence

import torch
import torch.nn as nn


class ChannelwiseActivation(nn.Module):
    def __init__(self, activations: Sequence[nn.Module]) -> None:
        super().__init__()
        self.activations = activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.split(1, dim=1)
        activations = [a(c) for a, c in zip(self.activations, channels)]
        return torch.cat(activations, dim=1)
