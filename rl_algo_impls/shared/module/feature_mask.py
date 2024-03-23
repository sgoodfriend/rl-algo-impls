import torch
import torch.nn as nn


class FeatureMask(nn.Module):
    def __init__(
        self,
        feature_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.feature_mask = feature_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.feature_mask]
