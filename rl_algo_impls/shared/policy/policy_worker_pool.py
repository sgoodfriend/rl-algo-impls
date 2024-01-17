import asyncio
from asyncio import Event
from collections import deque
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)

import numpy as np
import ray

from rl_algo_impls.shared.policy.abstract_policy import AbstractPolicy, Step
from rl_algo_impls.shared.policy.policy_actor import PolicyActor
from rl_algo_impls.wrappers.vector_wrapper import ObsType

if TYPE_CHECKING:
    from rl_algo_impls.shared.policy.policy import Policy
    from rl_algo_impls.shared.tensor_utils import NumpyOrDict


@dataclass
class ActorAssignment:
    event: Event
    actor: Optional[PolicyActor] = None


T = TypeVar("T")


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

        actors = [PolicyActor.remote(cuda_idx) for cuda_idx in _cuda_indexes]
        self.actor_by_id = {get_actor_id(a): a for a in actors}
        self._actor_ids_occupied: Set[str] = set()
        self.waiting_events_for_actor_id: Dict[str, Deque[Event]] = {
            a_id: deque() for a_id in self.actor_by_id
        }
        self.awaiting_assignment: Deque[ActorAssignment] = deque()

    def get_idle_actor(self) -> Optional[PolicyActor]:
        idle_actors = [
            actor
            for actor_id, actor in self.actor_by_id.items()
            if actor_id not in self._actor_ids_occupied
        ]
        return idle_actors[0] if idle_actors else None

    async def execute_task_once(
        self, task: Callable[[PolicyActor], Coroutine[Any, Any, T]]
    ) -> T:
        a = self.get_idle_actor()
        if a:
            return await self._execute_task_on_actor(task, a, True)
        aa = ActorAssignment(Event())
        self.awaiting_assignment.append(aa)
        await aa.event.wait()
        assert aa.actor is not None
        return await self._execute_task_on_actor(task, aa.actor, False)

    async def _execute_task_on_actor(
        self,
        task: Callable[[PolicyActor], Coroutine[Any, Any, T]],
        actor: PolicyActor,
        wait_if_actor_occupied: bool,
    ) -> T:
        actor_id = get_actor_id(actor)
        if wait_if_actor_occupied and actor_id in self._actor_ids_occupied:
            event = Event()
            self.waiting_events_for_actor_id[actor_id].append(event)
            await event.wait()
        self._actor_ids_occupied.add(actor_id)
        result = await task(actor)
        if self.waiting_events_for_actor_id[actor_id]:
            self.waiting_events_for_actor_id[actor_id].popleft().set()
        elif self.awaiting_assignment:
            assignment = self.awaiting_assignment.popleft()
            assignment.actor = actor
            assignment.event.set()
        else:
            self._actor_ids_occupied.remove(actor_id)
        return result

    async def execute_task_on_all(
        self, task: Callable[[PolicyActor], Coroutine]
    ) -> None:
        await asyncio.gather(
            *[
                self._execute_task_on_actor(task, a, True)
                for a in self.actor_by_id.values()
            ]
        )

    async def add_policy(self, policy_id: str, policy: "Policy") -> None:
        task = lambda a: a.add_policy.remote(policy_id, policy)
        await self.execute_task_on_all(task)

    async def clone_policy(
        self, origin_policy_id: str, destination_policy_id: str
    ) -> None:
        task = lambda a: a.clone_policy.remote(origin_policy_id, destination_policy_id)
        await self.execute_task_on_all(task)

    async def transfer_state(
        self,
        origin_policy_id: str,
        destination_policy_id: str,
        delete_origin_policy: bool = False,
    ) -> None:
        task = lambda a: a.transfer_state.remote(
            origin_policy_id,
            destination_policy_id,
            delete_origin_policy=delete_origin_policy,
        )
        await self.execute_task_on_all(task)

    async def act(
        self,
        policy_id: str,
        obs: ObsType,
        deterministic: bool = True,
        action_masks: Optional["NumpyOrDict"] = None,
    ) -> np.ndarray:
        task = lambda a: a.act.remote(
            policy_id, obs, deterministic, action_masks=action_masks
        )
        return await self.execute_task_once(task)

    async def reset_noise(self, policy_id: str) -> None:
        task = lambda a: a.reset_noise.remote(policy_id)
        await self.execute_task_on_all(task)

    async def save(self, policy_id: str, path: str) -> None:
        task = lambda a: a.save.remote(policy_id, path)
        await self.execute_task_once(task)

    async def set_state(self, policy_id: str, state: Any) -> None:
        task = lambda a: a.set_state.remote(policy_id, state)
        await self.execute_task_on_all(task)

    async def eval(self, policy_id: str) -> None:
        task = lambda a: a.eval.remote(policy_id)
        await self.execute_task_on_all(task)

    async def train(self, policy_id: str, mode: bool = True) -> None:
        task = lambda a: a.train.remote(policy_id, mode)
        await self.execute_task_on_all(task)

    async def value(self, policy_id: str, obs: ObsType) -> np.ndarray:
        task = lambda a: a.value.remote(policy_id, obs)
        return await self.execute_task_once(task)

    async def step(
        self, policy_id: str, obs: ObsType, action_masks: Optional["NumpyOrDict"] = None
    ) -> Step:
        task = lambda a: a.step.remote(policy_id, obs, action_masks=action_masks)
        return await self.execute_task_once(task)

    async def logprobs(
        self,
        policy_id: str,
        obs: ObsType,
        actions: "NumpyOrDict",
        action_masks: Optional["NumpyOrDict"] = None,
    ) -> np.ndarray:
        task = lambda a: a.logprobs.remote(
            policy_id, obs, actions, action_masks=action_masks
        )
        return await self.execute_task_once(task)


def get_actor_id(
    actor: PolicyActor,  # type: ignore
) -> str:
    return actor._actor_id.hex()
