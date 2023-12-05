import torch
import torch.nn as nn


class ChannelLayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
