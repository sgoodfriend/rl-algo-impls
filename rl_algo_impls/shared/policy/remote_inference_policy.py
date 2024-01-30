import os
import uuid
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

import numpy as np
import ray

from rl_algo_impls.shared.policy.abstract_policy import AbstractPolicy, Step
from rl_algo_impls.shared.policy.policy_actor import PolicyActor
from rl_algo_impls.shared.policy.policy_actor_pool import PolicyActorPool
from rl_algo_impls.wrappers.vector_wrapper import ObsType

if TYPE_CHECKING:
    from rl_algo_impls.shared.policy.policy import Policy
    from rl_algo_impls.shared.tensor_utils import NumpyOrDict

RemoteInferencePolicySelf = TypeVar(
    "RemoteInferencePolicySelf", bound="RemoteInferencePolicy"
)

_current_actor_id: Optional[str] = None
_policy_worker_for_current_actor: Optional[PolicyActor] = None


class RemoteInferencePolicy(AbstractPolicy, Generic[ObsType]):
    def __init__(
        self,
        policy_actor_pool: PolicyActorPool,
        policy: "Policy",  # policy is None when using private cloning path
        **kwargs,
    ) -> None:
        self.policy_actor_pool = policy_actor_pool
        self.policy_id = uuid.uuid4().hex

        self.all_actors = ray.get(self.policy_actor_pool.get_all_actors.remote())

        _clone_origin = kwargs.get("_clone_origin", None)
        # Private cloning path
        if _clone_origin:
            assert policy is None
            assert (
                policy_actor_pool._actor_id == _clone_origin.policy_actor_pool._actor_id
            ), f"Cannot clone policy from different PolicyWorkerPool"
            self.requires_to_cpu = _clone_origin.requires_to_cpu
            ray.get(
                [
                    actor.clone_policy.remote(_clone_origin.policy_id, self.policy_id)
                    for actor in self.all_actors
                ]
            )
        # Public policy path
        else:
            self.requires_to_cpu = bool(policy.device.index)
            if self.requires_to_cpu:
                import torch

                policy = deepcopy(policy).to(torch.device("cpu"))
            ray.get(
                [
                    actor.add_policy.remote(self.policy_id, policy)
                    for actor in self.all_actors
                ]
            )

    def __repr__(self) -> str:
        return super().__repr__() + f"(policy_id={self.policy_id})"

    @property
    def assigned_policy_actor(self) -> PolicyActor:
        global _current_actor_id
        global _policy_worker_for_current_actor
        if _current_actor_id is None:
            _current_actor_id = ray.get_runtime_context().get_actor_id()
        if _policy_worker_for_current_actor is None:
            try:
                cuda_visible_devices = list(
                    map(
                        int,
                        (
                            s
                            for s in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(
                                ","
                            )
                            if s
                        ),
                    )
                )
            except ValueError:
                warnings.warn(
                    f"Invalid CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}"
                )
                cuda_visible_devices = []
            _policy_worker_for_current_actor = ray.get(
                self.policy_actor_pool.get_policy_for_actor_id.remote(
                    _current_actor_id,
                    cuda_visible_devices[0] if len(cuda_visible_devices) else None,
                )
            )
        return _policy_worker_for_current_actor

    def act(
        self,
        obs: ObsType,
        deterministic: bool = True,
        action_masks: Optional["NumpyOrDict"] = None,
    ) -> np.ndarray:
        return ray.get(
            self.assigned_policy_actor.act.remote(
                self.policy_id, obs, deterministic, action_masks
            )
        )

    def reset_noise(self) -> None:
        for actor in self.all_actors:
            actor.reset_noise.remote(self.policy_id)

    def save(self, path: str) -> None:
        ray.get(self.assigned_policy_actor.save.remote(self.policy_id, path))

    def set_state(self, state: Any) -> None:
        if self.requires_to_cpu:
            state = {k: v.cpu() for k, v in state.items()}
        for actor in self.all_actors:
            actor.set_state.remote(self.policy_id, state)

    def eval(self: RemoteInferencePolicySelf) -> RemoteInferencePolicySelf:
        # No-op because policy is always in eval mode
        return self

    def train(
        self: RemoteInferencePolicySelf, mode: bool = True
    ) -> RemoteInferencePolicySelf:
        # No-op because we don't want to train the policy remotely.
        return self

    def value(self, obs: ObsType) -> np.ndarray:
        return ray.get(self.assigned_policy_actor.value.remote(self.policy_id, obs))

    def step(self, obs: ObsType, action_masks: Optional["NumpyOrDict"] = None) -> Step:
        return ray.get(
            self.assigned_policy_actor.step.remote(self.policy_id, obs, action_masks)
        )

    def logprobs(
        self,
        obs: ObsType,
        actions: "NumpyOrDict",
        action_masks: Optional["NumpyOrDict"] = None,
    ) -> np.ndarray:
        return ray.get(
            self.assigned_policy_actor.logprobs.remote(
                self.policy_id, obs, actions, action_masks
            )
        )

    def clone(self: RemoteInferencePolicySelf) -> RemoteInferencePolicySelf:
        return self.__class__(
            self.policy_actor_pool,
            None,  # type: ignore
            _clone_origin=self,
        )

    def transfer_policy_to(
        self: RemoteInferencePolicySelf,
        target: RemoteInferencePolicySelf,
        delete_origin_policy: bool = False,
    ) -> None:
        assert (
            self.policy_actor_pool._actor_id == target.policy_actor_pool._actor_id
        ), f"Cannot currently transfer policy between different PolicyWorkerPools"
        for agent in self.all_actors:
            agent.transfer_state.remote(
                self.policy_id,
                target.policy_id,
                delete_origin_policy=delete_origin_policy,
            )
