from asyncio import Event
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Union

import numpy as np
import ray
from ray.util.queue import Queue as RayQueue

from rl_algo_impls.shared.policy.abstract_policy import AbstractPolicy, Step
from rl_algo_impls.shared.policy.policy_worker import PolicyWorker, PolicyWorkerJob
from rl_algo_impls.wrappers.vector_wrapper import ObsType

if TYPE_CHECKING:
    from rl_algo_impls.shared.policy.policy import Policy
    from rl_algo_impls.shared.tensor_utils import NumpyOrDict


@dataclass
class WorkerState:
    worker: PolicyWorker
    queue: RayQueue
    idle: bool = True


@ray.remote
class PolicyWorkerPool(AbstractPolicy, Generic[ObsType]):
    def __init__(
        self, num_policy_workers: int, cuda_indexes: Union[int, List[int], None]
    ) -> None:
        if isinstance(cuda_indexes, (int, type(None))):
            _cuda_indexes = [cuda_indexes] * num_policy_workers
        elif len(cuda_indexes) == 1:
            _cuda_indexes = cuda_indexes * num_policy_workers
        else:
            _cuda_indexes = cuda_indexes
        assert len(_cuda_indexes) == num_policy_workers

        self.shared_job_queue = RayQueue()

        self.worker_state_by_id = {}
        for cuda_idx in _cuda_indexes:
            queue = RayQueue()
            worker = PolicyWorker.remote(
                cuda_idx,
                ray.get_runtime_context().current_actor,
                queue,
                self.shared_job_queue,
            )
            worker_state = WorkerState(worker, queue)
            self.worker_state_by_id[worker._actor_id.hex()] = worker_state

        self._next_job_id = 0
        self.job_events_by_id: Dict[int, Event] = {}
        self.job_results_by_id: Dict[int, Any] = {}

    def next_job_id(self) -> int:
        job_id = self._next_job_id
        self._next_job_id += 1
        return job_id

    def _submit_job_to_all_workers(self, job: PolicyWorkerJob) -> None:
        for worker_state in self.worker_state_by_id.values():
            worker_state.queue.put(job)
            if worker_state.idle:
                worker_state.idle = False
                worker_state.worker.run.remote()

    def _submit_job(self, job: PolicyWorkerJob) -> None:
        assert not job.wait_for_output, "Use _execute_job instead"
        self.shared_job_queue.put(job)
        self.run_idle_worker()

    async def _execute_job(self, job: PolicyWorkerJob) -> Any:
        assert job.wait_for_output, "Use _submit_job instead"
        event = Event()
        self.job_events_by_id[job.job_id] = event
        self.shared_job_queue.put(job)
        self.run_idle_worker()
        await event.wait()
        del self.job_events_by_id[job.job_id]
        return self.job_results_by_id.pop(job.job_id)

    def job_completed(self, job_id: int, result: Any) -> None:
        self.job_results_by_id[job_id] = result
        if job_id in self.job_events_by_id:
            self.job_events_by_id[job_id].set()

    def worker_done(self, worker_actor_id: str) -> None:
        worker_state = self.worker_state_by_id[worker_actor_id]
        if worker_state.queue.empty() and self.shared_job_queue.empty():
            worker_state.idle = True
        else:
            worker_state.worker.run.remote()

    def run_idle_worker(self) -> None:
        for worker_state in self.worker_state_by_id.values():
            if worker_state.idle:
                worker_state.idle = False
                worker_state.worker.run.remote()
                break

    def add_policy(self, policy_id: str, policy: "Policy") -> None:
        job = PolicyWorkerJob(
            "add_policy", (policy_id, policy), self.next_job_id(), False
        )
        self._submit_job_to_all_workers(job)

    def clone_policy(self, origin_policy_id: str, destination_policy_id: str) -> None:
        job = PolicyWorkerJob(
            "clone_policy",
            (origin_policy_id, destination_policy_id),
            self.next_job_id(),
            False,
        )
        self._submit_job_to_all_workers(job)

    def transfer_state(
        self,
        origin_policy_id: str,
        destination_policy_id: str,
        delete_origin_policy: bool = False,
    ) -> None:
        job = PolicyWorkerJob(
            "transfer_state",
            (origin_policy_id, destination_policy_id, delete_origin_policy),
            self.next_job_id(),
            False,
        )
        self._submit_job_to_all_workers(job)

    async def act(
        self,
        policy_id: str,
        obs: ObsType,
        deterministic: bool = True,
        action_masks: Optional["NumpyOrDict"] = None,
    ) -> np.ndarray:
        job = PolicyWorkerJob(
            "act",
            (policy_id, obs, deterministic, action_masks),
            self.next_job_id(),
            True,
        )
        return await self._execute_job(job)

    def reset_noise(self, policy_id: str) -> None:
        job = PolicyWorkerJob("reset_noise", (policy_id,), self.next_job_id(), False)
        self._submit_job_to_all_workers(job)

    def save(self, policy_id: str, path: str) -> None:
        job = PolicyWorkerJob("save", (policy_id, path), self.next_job_id(), False)
        self._submit_job(job)

    def set_state(self, policy_id: str, state: Any) -> None:
        job = PolicyWorkerJob(
            "set_state", (policy_id, state), self.next_job_id(), False
        )
        self._submit_job_to_all_workers(job)

    def eval(self, policy_id: str) -> None:
        job = PolicyWorkerJob("eval", (policy_id,), self.next_job_id(), False)
        self._submit_job_to_all_workers(job)

    def train(self, policy_id: str, mode: bool = True) -> None:
        job = PolicyWorkerJob("train", (policy_id, mode), self.next_job_id(), False)
        self._submit_job_to_all_workers(job)

    async def value(self, policy_id: str, obs: ObsType) -> np.ndarray:
        job = PolicyWorkerJob("value", (policy_id, obs), self.next_job_id(), True)
        return await self._execute_job(job)

    async def step(
        self, policy_id: str, obs: ObsType, action_masks: Optional["NumpyOrDict"] = None
    ) -> Step:
        job = PolicyWorkerJob(
            "step", (policy_id, obs, action_masks), self.next_job_id(), True
        )
        return await self._execute_job(job)

    async def logprobs(
        self,
        policy_id: str,
        obs: ObsType,
        actions: "NumpyOrDict",
        action_masks: Optional["NumpyOrDict"] = None,
    ) -> np.ndarray:
        job = PolicyWorkerJob(
            "logprobs",
            (policy_id, obs, actions, action_masks),
            self.next_job_id(),
            True,
        )
        return await self._execute_job(job)
