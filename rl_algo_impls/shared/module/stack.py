import torch
import torch.nn as nn


class HStack(nn.ModuleList):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.hstack([m(x) for m in self])


class Stack(nn.ModuleList):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self], dim=1)
