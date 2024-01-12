from typing import TYPE_CHECKING, Any, Generic, Optional

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

        self._policy: Optional["Policy"] = None

    def set_policy(self, policy: "Policy", train: Optional[bool] = None) -> None:
        self._policy = policy
        if train is not None:
            self._policy.train(train)

    def act(
        self,
        obs: ObsType,
        deterministic: bool = True,
        action_masks: Optional["NumpyOrDict"] = None,
    ) -> np.ndarray:
        assert self._policy is not None, "Policy must be set"
        return self._policy.act(obs, deterministic, action_masks)

    def reset_noise(self) -> None:
        assert self._policy is not None, "Policy must be set"
        self._policy.reset_noise()

    def save(self, path: str) -> None:
        assert self._policy is not None, "Policy must be set"
        self._policy.save(path)

    def set_state(self, state: Any) -> None:
        assert self._policy is not None, "Policy must be set"
        self._policy.set_state(state)

    def eval(self) -> None:
        assert self._policy is not None, "Policy must be set"
        self._policy.eval()

    def train(self, mode: bool = True) -> None:
        assert self._policy is not None, "Policy must be set"
        self._policy.train(mode)

    def value(self, obs: Any) -> np.ndarray:
        assert self._policy is not None, "Policy must be set"
        return self._policy.value(obs)

    def step(self, obs: Any, action_masks: Optional["NumpyOrDict"] = None) -> Step:
        assert self._policy is not None, "Policy must be set"
        return self._policy.step(obs, action_masks)

    def transfer_policy_to(
        self, target: "PolicyActor", exit_after_transfer: bool = False
    ) -> None:
        assert self._policy is not None, "Policy must be set"
        ray.get(target.set_policy.remote(self._policy))
        if exit_after_transfer:
            ray.actor.exit_actor()
