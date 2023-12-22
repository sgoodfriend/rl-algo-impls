import os
from abc import ABC, abstractmethod
from typing import Dict, Generic, NamedTuple, Optional, Type, TypeVar, Union

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from rl_algo_impls.shared.tensor_utils import NumpyOrDict, TensorOrDict, numpy_to_tensor
from rl_algo_impls.shared.trackable import Trackable
from rl_algo_impls.wrappers.vector_wrapper import ObsType, VectorEnv

ACTIVATION: Dict[str, Type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "identity": nn.Identity,
    "sigmoid": nn.Sigmoid,
}

MODEL_FILENAME = "model.pth"

EnvSpacesSelf = TypeVar("EnvSpacesSelf", bound="EnvSpaces")


class EnvSpaces(NamedTuple):
    single_observation_space: gymnasium.Space
    single_action_space: gymnasium.Space
    action_plane_space: Optional[gymnasium.Space]
    num_envs: int

    @classmethod
    def from_vec_env(cls: Type[EnvSpacesSelf], env: VectorEnv) -> EnvSpacesSelf:
        return cls(
            single_observation_space=env.single_observation_space,
            single_action_space=env.single_action_space,
            action_plane_space=getattr(env, "action_plane_space", None),
            num_envs=env.num_envs,
        )


PolicySelf = TypeVar("PolicySelf", bound="Policy")


class Policy(nn.Module, Trackable, ABC, Generic[ObsType]):
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

    @abstractmethod
    def act(
        self,
        obs: ObsType,
        deterministic: bool = True,
        action_masks: Optional[NumpyOrDict] = None,
    ) -> np.ndarray:
        ...

    def save_weights(self, path: str) -> None:
        torch.save(
            self.state_dict(),
            os.path.join(path, MODEL_FILENAME),
        )

    def load_weights(self, path: str) -> None:
        self.load_state_dict(
            torch.load(os.path.join(path, MODEL_FILENAME), map_location=self._device)
        )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.save_weights(path)

    def load(self, path: str) -> None:
        self.load_weights(path)

    def reset_noise(self) -> None:
        pass

    def _as_tensor(self, a: NumpyOrDict) -> TensorOrDict:
        assert self._device
        return numpy_to_tensor(a, self._device)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
