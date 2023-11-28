from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(
        self,
        pool_output_size: Union[int, None, Tuple[Optional[int], Optional[int]]] = 1,
    ) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(pool_output_size)
        self.max_pool = nn.AdaptiveMaxPool2d(pool_output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pooled = self.avg_pool(x)
        max_pooled = self.max_pool(x)
        return torch.cat([avg_pooled, max_pooled], dim=1)
