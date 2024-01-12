from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Generic, Optional

import numpy as np
import ray

from rl_algo_impls.shared.policy.abstract_policy import AbstractPolicy, Step
from rl_algo_impls.wrappers.vector_wrapper import ObsType

if TYPE_CHECKING:
    from rl_algo_impls.shared.policy.policy import Policy
    from rl_algo_impls.shared.tensor_utils import NumpyOrDict


@ray.remote
class PolicyActor(AbstractPolicy, Generic[ObsType]):
    def __init__(self, cuda_index: Optional[int]) -> None:
        from rl_algo_impls.utils.ray import init_ray_actor

        init_ray_actor(
            cuda_visible_devices=[cuda_index] if cuda_index is not None else []
        )

        self._next_idx = 0
        self._policies_by_key: Dict[int, "Policy"] = {}

    def add_policy(self, policy: "Policy") -> int:
        idx = self._next_idx
        self._next_idx += 1
        self._policies_by_key[idx] = policy
        return idx

    def clone_policy(self, policy_idx: int) -> int:
        assert policy_idx in self._policies_by_key, f"No policy with idx {policy_idx}"
        return self.add_policy(deepcopy(self._policies_by_key[policy_idx]))

    def transfer_state(
        self,
        origin_policy_idx: int,
        destination_policy_idx: int,
        delete_origin_policy: bool = False,
    ) -> None:
        assert (
            origin_policy_idx in self._policies_by_key
        ), f"No policy with idx {origin_policy_idx}"
        assert (
            destination_policy_idx in self._policies_by_key
        ), f"No policy with idx {destination_policy_idx}"
        self._policies_by_key[destination_policy_idx].set_state(
            self._policies_by_key[origin_policy_idx].get_state()
        )
        if delete_origin_policy:
            del self._policies_by_key[origin_policy_idx]

    def act(
        self,
        policy_idx: int,
        obs: ObsType,
        deterministic: bool = True,
        action_masks: Optional["NumpyOrDict"] = None,
    ) -> np.ndarray:
        assert policy_idx in self._policies_by_key, f"No policy with idx {policy_idx}"
        return self._policies_by_key[policy_idx].act(obs, deterministic, action_masks)

    def reset_noise(self, policy_idx: int) -> None:
        assert policy_idx in self._policies_by_key, f"No policy with idx {policy_idx}"
        self._policies_by_key[policy_idx].reset_noise()

    def save(self, policy_idx: int, path: str) -> None:
        assert policy_idx in self._policies_by_key, f"No policy with idx {policy_idx}"
        self._policies_by_key[policy_idx].save(path)

    def set_state(self, policy_idx: int, state: Any) -> None:
        assert policy_idx in self._policies_by_key, f"No policy with idx {policy_idx}"
        self._policies_by_key[policy_idx].set_state(state)

    def eval(self, policy_idx: int) -> None:
        assert policy_idx in self._policies_by_key, f"No policy with idx {policy_idx}"
        self._policies_by_key[policy_idx].eval()

    def train(self, policy_idx: int, mode: bool = True) -> None:
        assert policy_idx in self._policies_by_key, f"No policy with idx {policy_idx}"
        self._policies_by_key[policy_idx].train(mode)

    def value(self, policy_idx: int, obs: Any) -> np.ndarray:
        assert policy_idx in self._policies_by_key, f"No policy with idx {policy_idx}"
        return self._policies_by_key[policy_idx].value(obs)

    def step(
        self, policy_idx: int, obs: Any, action_masks: Optional["NumpyOrDict"] = None
    ) -> Step:
        assert policy_idx in self._policies_by_key, f"No policy with idx {policy_idx}"
        return self._policies_by_key[policy_idx].step(obs, action_masks)

    def transfer_policy_to(
        self,
        origin_policy_idx: int,
        target: "PolicyActor",
        delete_origin_policy: bool = False,
    ) -> int:
        assert (
            origin_policy_idx in self._policies_by_key
        ), f"No policy with idx {origin_policy_idx}"
        destination_policy_idx = ray.get(
            target.add_policy.remote(self._policies_by_key[origin_policy_idx])
        )
        if delete_origin_policy:
            del self._policies_by_key[origin_policy_idx]
        return destination_policy_idx
