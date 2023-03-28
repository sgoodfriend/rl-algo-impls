from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution


class PiForward(NamedTuple):
    pi: Distribution
    logp_a: Optional[torch.Tensor]
    entropy: Optional[torch.Tensor]


class Actor(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
    ) -> PiForward:
        ...

    def sample_weights(self, batch_size: int = 1) -> None:
        pass

    @property
    @abstractmethod
    def action_shape(self) -> Tuple[int, ...]:
        ...
