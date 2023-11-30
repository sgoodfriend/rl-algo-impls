import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn

from rl_algo_impls.shared.tensor_utils import NumpyOrDict, TensorOrDict, numpy_to_tensor
from rl_algo_impls.wrappers.normalize import NormalizeObservation, NormalizeReward
from rl_algo_impls.wrappers.vector_wrapper import ObsType, VectorEnv, find_wrapper

ACTIVATION: Dict[str, Type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "identity": nn.Identity,
    "sigmoid": nn.Sigmoid,
}

VEC_NORMALIZE_FILENAME = "vecnormalize.pkl"
MODEL_FILENAME = "model.pth"
NORMALIZE_OBSERVATION_FILENAME = "norm_obs.npz"
NORMALIZE_REWARD_FILENAME = "norm_reward.npz"

PolicySelf = TypeVar("PolicySelf", bound="Policy")


class Policy(nn.Module, ABC, Generic[ObsType]):
    @abstractmethod
    def __init__(self, env: VectorEnv, **kwargs) -> None:
        super().__init__()
        self.env = env
        norm_observation = find_wrapper(env, NormalizeObservation)
        self.norm_observation_rms = norm_observation.rms if norm_observation else None
        norm_reward = find_wrapper(env, NormalizeReward)
        self.norm_reward_rms = norm_reward.rms if norm_reward else None
        self.device = None

    def to(
        self: PolicySelf,
        device: Optional[torch.device] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
        non_blocking: bool = False,
    ) -> PolicySelf:
        super().to(device, dtype, non_blocking)
        self.device = device
        return self

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
            torch.load(os.path.join(path, MODEL_FILENAME), map_location=self.device)
        )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        if self.norm_observation_rms:
            self.norm_observation_rms.save(
                os.path.join(path, NORMALIZE_OBSERVATION_FILENAME)
            )
        if self.norm_reward_rms:
            self.norm_reward_rms.save(os.path.join(path, NORMALIZE_REWARD_FILENAME))
        self.save_weights(path)

    def load(
        self, path: str, load_norm_rms_count_override: Optional[int] = None
    ) -> None:
        self.load_weights(path)
        if self.norm_observation_rms:
            self.norm_observation_rms.load(
                os.path.join(path, NORMALIZE_OBSERVATION_FILENAME),
                count_override=load_norm_rms_count_override,
            )
        if self.norm_reward_rms:
            self.norm_reward_rms.load(
                os.path.join(path, NORMALIZE_REWARD_FILENAME),
                count_override=load_norm_rms_count_override,
            )

    def load_from(self: PolicySelf, policy: PolicySelf) -> PolicySelf:
        self.load_state_dict(policy.state_dict())
        if self.norm_observation_rms:
            assert policy.norm_observation_rms
            self.norm_observation_rms.load_from(policy.norm_observation_rms)
        if self.norm_reward_rms:
            assert policy.norm_reward_rms
            assert type(self.norm_reward_rms) == type(policy.norm_reward_rms)
            self.norm_reward_rms.load_from(policy.norm_reward_rms)  # type: ignore
        return self

    def __deepcopy__(self: PolicySelf, memo: Dict[int, Any]) -> PolicySelf:
        cls = self.__class__
        cpy = cls.__new__(cls)

        memo[id(self)] = cpy

        for k, v in self.__dict__.items():
            if k == "env":
                setattr(cpy, k, v)  # Don't deepcopy Env
            else:
                setattr(cpy, k, deepcopy(v, memo))

        return cpy

    def reset_noise(self) -> None:
        pass

    def _as_tensor(self, a: NumpyOrDict) -> TensorOrDict:
        assert self.device
        return numpy_to_tensor(a, self.device)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def sync_normalization(self, destination_env) -> None:
        current = destination_env
        while current != current.unwrapped:
            if isinstance(current, NormalizeObservation):
                assert self.norm_observation_rms
                current.rms = deepcopy(self.norm_observation_rms)
            elif isinstance(current, NormalizeReward):
                assert self.norm_reward_rms
                current.rms = deepcopy(self.norm_reward_rms)
            current = getattr(current, "env", current)
            if not current:
                raise AttributeError(f"{type(current)} doesn't include env attribute")
