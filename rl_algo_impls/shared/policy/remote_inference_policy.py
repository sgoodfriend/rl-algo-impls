import logging
import uuid
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

import numpy as np
import ray

from rl_algo_impls.shared.policy.abstract_policy import AbstractPolicy, Step
from rl_algo_impls.shared.policy.policy_worker_pool import (
    PolicyArgsRef,
    PolicyWorkerPool,
)
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
        policy_worker_pool: PolicyWorkerPool,
        policy: "Policy",  # policy is None when using private cloning path
        **kwargs,
    ) -> None:
        self.policy_worker_pool = policy_worker_pool
        self.policy_id = uuid.uuid4().hex

        _clone_origin = kwargs.get("_clone_origin", None)
        # Private cloning path
        if _clone_origin:
            assert policy is None
            assert (
                policy_worker_pool._actor_id
                == _clone_origin.policy_worker_pool._actor_id
            ), f"Cannot clone policy from different PolicyWorkerPool"
            ray.get(
                policy_worker_pool.clone_policy.remote(
                    _clone_origin.policy_id, self.policy_id
                )
            )
        # Public policy path
        else:
            ray.get(
                policy_worker_pool.add_policy.remote(
                    PolicyArgsRef(ray.put((self.policy_id, policy)))
                )
            )

    def __repr__(self) -> str:
        return super().__repr__() + f"(policy_id={self.policy_id})"

    def act(
        self,
        obs: ObsType,
        deterministic: bool = True,
        action_masks: Optional["NumpyOrDict"] = None,
    ) -> np.ndarray:
        return ray.get(
            self.policy_worker_pool.act.remote(
                PolicyArgsRef(
                    ray.put((self.policy_id, obs, deterministic, action_masks))
                )
            )
        )

    def reset_noise(self) -> None:
        return self.policy_worker_pool.reset_noise.remote(self.policy_id)

    def save(self, path: str) -> None:
        return self.policy_worker_pool.save.remote(self.policy_id, path)

    def set_state(self, state: Any) -> None:
        self.policy_worker_pool.set_state.remote(
            PolicyArgsRef(ray.put((self.policy_id, state)))
        )

    def eval(self: RemoteInferencePolicySelf) -> RemoteInferencePolicySelf:
        # No-op because policy is always in eval mode
        return self

    def train(
        self: RemoteInferencePolicySelf, mode: bool = True
    ) -> RemoteInferencePolicySelf:
        # No-op because we don't want to train the policy remotely.
        return self

    def value(self, obs: ObsType) -> np.ndarray:
        return ray.get(
            self.policy_worker_pool.value.remote(
                PolicyArgsRef(ray.put((self.policy_id, obs)))
            )
        )

    def step(self, obs: ObsType, action_masks: Optional["NumpyOrDict"] = None) -> Step:
        return ray.get(
            self.policy_worker_pool.step.remote(
                PolicyArgsRef(ray.put((self.policy_id, obs, action_masks)))
            )
        )

    def logprobs(
        self,
        obs: ObsType,
        actions: "NumpyOrDict",
        action_masks: Optional["NumpyOrDict"] = None,
    ) -> np.ndarray:
        return ray.get(
            self.policy_worker_pool.logprobs.remote(
                PolicyArgsRef(ray.put((self.policy_id, obs, actions, action_masks)))
            )
        )

    def clone(self: RemoteInferencePolicySelf) -> RemoteInferencePolicySelf:
        return self.__class__(
            self.policy_worker_pool,
            None,  # type: ignore
            _clone_origin=self,
        )

    def transfer_policy_to(
        self: RemoteInferencePolicySelf,
        target: RemoteInferencePolicySelf,
        delete_origin_policy: bool = False,
    ) -> None:
        assert (
            self.policy_worker_pool._actor_id == target.policy_worker_pool._actor_id
        ), f"Cannot currently transfer policy between different PolicyWorkerPools"
        self.policy_worker_pool.transfer_state.remote(
            self.policy_id,
            target.policy_id,
            delete_origin_policy=delete_origin_policy,
        )
