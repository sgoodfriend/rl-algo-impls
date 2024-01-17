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
class PolicyActor:
    def __init__(self, cuda_index: Optional[int]) -> None:
        from rl_algo_impls.utils.ray import init_ray_actor

        init_ray_actor(
            cuda_visible_devices=[cuda_index] if cuda_index is not None else []
        )

        self.policies_by_id: Dict[str, "Policy"] = {}

    def add_policy(self, args) -> None:
        policy_id, policy = args
        assert (
            policy_id not in self.policies_by_id
        ), f"Policy with id {policy_id} exists"
        self.policies_by_id[policy_id] = policy

    def clone_policy(self, origin_policy_id: str, destination_policy_id: str) -> None:
        assert (
            origin_policy_id in self.policies_by_id
        ), f"No policy with id {origin_policy_id}"
        assert (
            destination_policy_id not in self.policies_by_id
        ), f"Policy with id {destination_policy_id} exists"
        self.policies_by_id[destination_policy_id] = deepcopy(
            self.policies_by_id[origin_policy_id]
        )

    def transfer_state(
        self,
        origin_policy_id: str,
        destination_policy_id: str,
        delete_origin_policy: bool = False,
    ) -> None:
        assert (
            origin_policy_id in self.policies_by_id
        ), f"No policy with id {origin_policy_id}"
        assert (
            destination_policy_id in self.policies_by_id
        ), f"No policy with id {destination_policy_id}"
        self.policies_by_id[destination_policy_id].set_state(
            self.policies_by_id[origin_policy_id].get_state()
        )
        if delete_origin_policy:
            del self.policies_by_id[origin_policy_id]

    def act(
        self,
        args,
    ) -> np.ndarray:
        policy_id, obs, deterministic, action_masks = args
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        return self.policies_by_id[policy_id].act(obs, deterministic, action_masks)

    def reset_noise(self, policy_id: str) -> None:
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        self.policies_by_id[policy_id].reset_noise()

    def save(self, policy_id: str, path: str) -> None:
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        self.policies_by_id[policy_id].save(path)

    def set_state(self, args) -> None:
        policy_id, state = args
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        self.policies_by_id[policy_id].set_state(state)

    def eval(self, policy_id: str) -> None:
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        self.policies_by_id[policy_id].eval()

    def train(self, policy_id: str, mode: bool = True) -> None:
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        self.policies_by_id[policy_id].train(mode)

    def value(self, args) -> np.ndarray:
        policy_id, obs = args
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        return self.policies_by_id[policy_id].value(obs)

    def step(self, args) -> Step:
        policy_id, obs, action_masks = args
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        return self.policies_by_id[policy_id].step(obs, action_masks)

    def logprobs(self, args) -> np.ndarray:
        policy_id, obs, actions, action_masks = args
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        return self.policies_by_id[policy_id].logprobs(obs, actions, action_masks)
