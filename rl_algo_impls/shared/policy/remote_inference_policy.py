from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

import numpy as np
import ray

from rl_algo_impls.shared.policy.abstract_policy import AbstractPolicy, Step
from rl_algo_impls.shared.policy.policy_actor import PolicyActor
from rl_algo_impls.wrappers.vector_wrapper import ObsType

if TYPE_CHECKING:
    from rl_algo_impls.shared.policy.policy import Policy
    from rl_algo_impls.shared.tensor_utils import NumpyOrDict

RemoteInferencePolicySelf = TypeVar(
    "RemoteInferencePolicySelf", bound="RemoteInferencePolicy"
)


class RemoteInferencePolicy(AbstractPolicy, Generic[ObsType]):
    def __init__(
        self,
        cuda_index: Optional[int],
        policy: "Policy",  # policy is None when using private cloning path
        **kwargs,
    ) -> None:
        self.cuda_index = cuda_index

        _clone_origin = kwargs.get("_clone_origin", None)
        # Private cloning path
        if _clone_origin:
            if _clone_origin.cuda_index == cuda_index:
                self.policy_actor = _clone_origin.policy_actor
                self.policy_idx = self.policy_actor.clone_policy.remote(
                    _clone_origin.policy_idx
                )
            else:
                self.policy_actor = PolicyActor.remote(cuda_index)
                self.policy_idx = ray.get(
                    _clone_origin.transfer_policy_to.remote(
                        _clone_origin.policy_idx, self.policy_actor
                    )
                )
        # Public policy path
        else:
            self.policy_actor = PolicyActor.remote(cuda_index)
            self.policy_idx = ray.get(self.policy_actor.add_policy.remote(policy))

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f"(cuda_index={self.cuda_index}, policy_idx={self.policy_idx})"
        )

    def act(
        self,
        obs: ObsType,
        deterministic: bool = True,
        action_masks: Optional["NumpyOrDict"] = None,
    ) -> np.ndarray:
        return ray.get(
            self.policy_actor.act.remote(
                self.policy_idx, obs, deterministic, action_masks
            )
        )

    def reset_noise(self) -> None:
        return self.policy_actor.reset_noise.remote(self.policy_idx)

    def save(self, path: str) -> None:
        return self.policy_actor.save.remote(self.policy_idx, path)

    def set_state(self, state: Any) -> None:
        self.policy_actor.set_state.remote(self.policy_idx, state)

    def eval(self: RemoteInferencePolicySelf) -> RemoteInferencePolicySelf:
        # No-op because policy is always in eval mode
        return self

    def train(
        self: RemoteInferencePolicySelf, mode: bool = True
    ) -> RemoteInferencePolicySelf:
        # No-op because we don't want to train the policy remotely.
        return self

    def value(self, obs: ObsType) -> np.ndarray:
        return ray.get(self.policy_actor.value.remote(self.policy_idx, obs))

    def step(self, obs: ObsType, action_masks: Optional["NumpyOrDict"] = None) -> Step:
        return ray.get(
            self.policy_actor.step.remote(self.policy_idx, obs, action_masks)
        )

    def logprobs(
        self,
        obs: ObsType,
        actions: "NumpyOrDict",
        action_masks: Optional["NumpyOrDict"] = None,
    ) -> np.ndarray:
        return ray.get(
            self.policy_actor.logprobs.remote(
                self.policy_idx, obs, actions, action_masks
            )
        )

    def clone(
        self: RemoteInferencePolicySelf, target_cuda_index: Optional[int]
    ) -> RemoteInferencePolicySelf:
        return self.__class__(
            target_cuda_index,
            None,  # type: ignore
            _clone_origin=self,
        )

    def transfer_policy_to(
        self: RemoteInferencePolicySelf,
        target: RemoteInferencePolicySelf,
        delete_origin_policy: bool = False,
    ) -> None:
        if self.policy_actor._actor_id == target.policy_actor._actor_id:
            self.policy_actor.transfer_state.remote(
                self.policy_idx,
                target.policy_idx,
                delete_origin_policy=delete_origin_policy,
            )
        else:
            self.policy_actor.transfer_policy_to.remote(
                self.policy_idx,
                target.policy_actor,
                delete_origin_policy=delete_origin_policy,
            )
