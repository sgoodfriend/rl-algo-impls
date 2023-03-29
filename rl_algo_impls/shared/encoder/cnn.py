from abc import ABC, abstractmethod
from typing import Optional, Type

import torch.nn as nn


class CnnFeatureExtractor(nn.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        in_channels: int,
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__()
