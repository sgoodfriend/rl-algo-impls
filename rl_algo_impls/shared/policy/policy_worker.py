from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Generic, NamedTuple, Optional, Tuple

import numpy as np
import ray
import ray.util.queue
from ray.util.queue import Queue as RayQueue

from rl_algo_impls.shared.policy.abstract_policy import Step
from rl_algo_impls.wrappers.vector_wrapper import ObsType

if TYPE_CHECKING:
    from rl_algo_impls.shared.policy.policy import Policy
    from rl_algo_impls.shared.policy.policy_worker_pool import PolicyWorkerPool
    from rl_algo_impls.shared.tensor_utils import NumpyOrDict


class PolicyWorkerJob(NamedTuple):
    job_name: str
    args: Tuple[Any, ...]
    job_id: int
    wait_for_output: bool


@ray.remote
class PolicyWorker:
    def __init__(
        self,
        cuda_index: Optional[int],
        policy_worker_pool: "PolicyWorkerPool",
        worker_queue: RayQueue,
        shared_job_queue: RayQueue,
    ) -> None:
        from rl_algo_impls.utils.ray import init_ray_actor

        init_ray_actor(
            cuda_visible_devices=[cuda_index] if cuda_index is not None else []
        )

        self.worker_pool = policy_worker_pool
        self.worker_queue = worker_queue
        self.shared_job_queue = shared_job_queue

        self.policies_by_id: Dict[str, "Policy"] = {}

    def run(self) -> None:
        job = self.next_job()
        while job:
            self._run_job(job)
            job = self.next_job()
        self.worker_pool.worker_done.remote(ray.get_runtime_context().get_actor_id())

    def next_job(self) -> Optional[PolicyWorkerJob]:
        try:
            return self.worker_queue.get_nowait()
        except ray.util.queue.Empty:
            pass
        try:
            return self.shared_job_queue.get_nowait()
        except ray.util.queue.Empty:
            pass
        return None

    def _run_job(self, job: PolicyWorkerJob) -> None:
        job_name, args, job_id, wait_for_output = job
        fn = getattr(self, job_name)
        output = fn(*args)
        if wait_for_output:
            self.worker_pool.job_completed.remote(job_id, output)

    def add_policy(self, policy_id: str, policy: "Policy") -> None:
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
        policy_id: str,
        obs: ObsType,
        deterministic: bool = True,
        action_mask: Optional["NumpyOrDict"] = None,
    ) -> np.ndarray:
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        return self.policies_by_id[policy_id].act(obs, deterministic, action_mask)

    def reset_noise(self, policy_id: str) -> None:
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        self.policies_by_id[policy_id].reset_noise()

    def save(self, policy_id: str, path: str) -> None:
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        self.policies_by_id[policy_id].save(path)

    def set_state(self, policy_id: str, state: Any) -> None:
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        self.policies_by_id[policy_id].set_state(state)

    def eval(self, policy_id: str) -> None:
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        self.policies_by_id[policy_id].eval()

    def train(self, policy_id: str, mode: bool = True) -> None:
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        self.policies_by_id[policy_id].train(mode)

    def value(self, policy_id: str, obs: ObsType) -> np.ndarray:
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        return self.policies_by_id[policy_id].value(obs)

    def step(
        self,
        policy_id: str,
        obs: ObsType,
        action_mask: Optional["NumpyOrDict"] = None,
    ) -> Step:
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        return self.policies_by_id[policy_id].step(obs, action_mask)

    def logprobs(
        self,
        policy_id: str,
        obs: ObsType,
        actions: "NumpyOrDict",
        action_mask: Optional["NumpyOrDict"] = None,
    ) -> np.ndarray:
        assert policy_id in self.policies_by_id, f"No policy with id {policy_id}"
        return self.policies_by_id[policy_id].logprobs(obs, actions, action_mask)
