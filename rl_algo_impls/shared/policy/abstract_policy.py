from abc import ABC, abstractmethod
from typing import Any, Generic, NamedTuple, Optional, TypeVar

import numpy as np

from rl_algo_impls.shared.tensor_utils import NumpyOrDict
from rl_algo_impls.wrappers.vector_wrapper import ObsType


class Step(NamedTuple):
    a: NumpyOrDict
    v: np.ndarray
    logp_a: np.ndarray
    clamped_a: np.ndarray


AbstractPolicySelf = TypeVar("AbstractPolicySelf", bound="AbstractPolicy")


class AbstractPolicy(ABC, Generic[ObsType]):
    @abstractmethod
    def act(
        self,
        obs: ObsType,
        deterministic: bool = True,
        action_masks: Optional[NumpyOrDict] = None,
    ) -> np.ndarray:
        ...

    @abstractmethod
    def reset_noise(self) -> None:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def set_state(self, state: Any) -> None:
        ...

    @abstractmethod
    def eval(self: AbstractPolicySelf) -> AbstractPolicySelf:
        ...

    @abstractmethod
    def train(self: AbstractPolicySelf, mode: bool = True) -> AbstractPolicySelf:
        ...

    # OnPolicy methods
    @abstractmethod
    def value(self, obs: ObsType) -> np.ndarray:
        ...

    @abstractmethod
    def step(self, obs: ObsType, action_masks: Optional[NumpyOrDict] = None) -> Step:
        ...

    def logprobs(
        self,
        obs: ObsType,
        actions: NumpyOrDict,
        action_masks: Optional[NumpyOrDict] = None,
    ) -> np.ndarray:
        ...
