import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar, Union

import torch
import torch.nn as nn

from rl_algo_impls.shared.policy.abstract_policy import AbstractPolicy
from rl_algo_impls.shared.tensor_utils import NumpyOrDict, TensorOrDict, numpy_to_tensor
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces

ACTIVATION: Dict[str, Type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "identity": nn.Identity,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
}

MODEL_FILENAME = "model.pth"

PolicySelf = TypeVar("PolicySelf", bound="Policy")


class Policy(nn.Module, AbstractPolicy, ABC):
    @abstractmethod
    def __init__(
        self,
        env_spaces: EnvSpaces,
        **kwargs,
    ) -> None:
        super().__init__()
        self.env_spaces = env_spaces

        self._device = None

    def to(
        self: PolicySelf,
        device: Optional[torch.device] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
        non_blocking: bool = False,
    ) -> PolicySelf:
        super().to(device, dtype, non_blocking)
        self._device = device
        return self

    @property
    def device(self) -> torch.device:
        assert self._device, "Expect device to be set"
        return self._device

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.get_state(), os.path.join(path, MODEL_FILENAME))

    def load(self, path: str) -> None:
        self.set_state(
            torch.load(os.path.join(path, MODEL_FILENAME), map_location=self._device)
        )

    def get_state(self) -> Any:
        return self.state_dict()

    def set_state(self, state: Any) -> None:
        self.load_state_dict(state)

    def reset_noise(self) -> None:
        pass

    def _as_tensor(self, a: NumpyOrDict) -> TensorOrDict:
        assert self._device
        return numpy_to_tensor(a, self._device)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
