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
    def __init__(self, cuda_index: Optional[int], policy: Optional["Policy"]) -> None:
        self.policy_actor = PolicyActor.remote(cuda_index)
        if policy is not None:
            ray.get(self.policy_actor.set_policy.remote(policy, train=False))

    def act(
        self,
        obs: ObsType,
        deterministic: bool = True,
        action_masks: Optional["NumpyOrDict"] = None,
    ) -> np.ndarray:
        return ray.get(self.policy_actor.act.remote(obs, deterministic, action_masks))

    def reset_noise(self) -> None:
        return self.policy_actor.reset_noise.remote()

    def save(self, path: str) -> None:
        return self.policy_actor.save.remote(path)

    def set_state(self, state: Any) -> None:
        self.policy_actor.set_state.remote(state)

    def eval(self: RemoteInferencePolicySelf) -> RemoteInferencePolicySelf:
        # No-op because policy is always in eval mode
        return self

    def train(
        self: RemoteInferencePolicySelf, mode: bool = True
    ) -> RemoteInferencePolicySelf:
        # No-op because we don't want to train the policy remotely.
        return self

    def value(self, obs: Any) -> np.ndarray:
        return ray.get(self.policy_actor.value.remote(obs))

    def step(self, obs: Any, action_masks: Optional["NumpyOrDict"] = None) -> Step:
        return ray.get(self.policy_actor.step.remote(obs, action_masks))

    def clone(
        self: RemoteInferencePolicySelf, target_cuda_index: Optional[int]
    ) -> RemoteInferencePolicySelf:
        policy_clone = self.__class__(target_cuda_index, None)
        ray.get(self.policy_actor.transfer_policy_to.remote(policy_clone.policy_actor))
        return policy_clone

    def transfer_policy_to(
        self: RemoteInferencePolicySelf,
        target: RemoteInferencePolicySelf,
        exit_after_transfer: bool = False,
    ) -> None:
        ray.get(
            self.policy_actor.transfer_policy_to.remote(
                target.policy_actor, exit_after_transfer=exit_after_transfer
            )
        )
