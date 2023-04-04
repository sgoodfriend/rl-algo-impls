from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from gym.spaces import Box, Discrete, Space

from rl_algo_impls.shared.actor import PiForward


class ACNForward(NamedTuple):
    pi_forward: PiForward
    v: torch.Tensor


class ActorCriticNetwork(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> ACNForward:
        ...

    @abstractmethod
    def distribution_and_value(
        self, obs: torch.Tensor, action_masks: Optional[torch.Tensor] = None
    ) -> ACNForward:
        ...

    @abstractmethod
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def reset_noise(self, batch_size: Optional[int] = None) -> None:
        ...

    @property
    def action_shape(self) -> Tuple[int, ...]:
        ...


def default_hidden_sizes(obs_space: Space) -> Sequence[int]:
    if isinstance(obs_space, Box):
        if len(obs_space.shape) == 3:  # type: ignore
            # By default feature extractor to output has no hidden layers
            return []
        elif len(obs_space.shape) == 1:  # type: ignore
            return [64, 64]
        else:
            raise ValueError(f"Unsupported observation space: {obs_space}")
    elif isinstance(obs_space, Discrete):
        return [64]
    else:
        raise ValueError(f"Unsupported observation space: {obs_space}")
