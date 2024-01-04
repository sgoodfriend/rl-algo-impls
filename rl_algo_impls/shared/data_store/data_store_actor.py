import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Type, TypeVar

import ray

from rl_algo_impls.shared.data_store.algorithm_state import RemoteAlgorithmState
from rl_algo_impls.shared.data_store.data_store_data import (
    CheckpointState,
    DataStoreFinalization,
    EvalEnqueue,
    EvalView,
    LearnerView,
    RolloutUpdate,
    RolloutView,
)
from rl_algo_impls.shared.evaluator.abstract_evaluator import AbstractEvaluator
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.trackable import Trackable

if TYPE_CHECKING:
    from rl_algo_impls.rollout.rollout import Rollout
    from rl_algo_impls.shared.policy.policy import Policy


class RemoteLearnerInitializeData(NamedTuple):
    policy: "Policy"
    algo_state: Trackable
    load_path: Optional[str]


RemoteEvalEnqueueSelf = TypeVar("RemoteEvalEnqueueSelf", bound="RemoteEvalEnqueue")


class RemoteEvalEnqueue(NamedTuple):
    algo_state: Trackable

    @classmethod
    def from_eval_enqueue(
        cls: Type[RemoteEvalEnqueueSelf], eval_enqueue: Optional[EvalEnqueue]
    ) -> Optional[RemoteEvalEnqueueSelf]:
        return cls(RemoteAlgorithmState(eval_enqueue.algo)) if eval_enqueue else None


class RemoteLearnerUpdate(NamedTuple):
    policy: "Policy"
    rollout_params: Dict[str, Any]
    timesteps_elapsed: int
    eval_enqueue: Optional[RemoteEvalEnqueue]


@ray.remote
class DataStoreActor:
    def __init__(self, history_size: int) -> None:
        self.history_size = history_size

        self.is_closed = False
        self.timesteps_elapsed = 0
        self.latest_policy: Optional["Policy"] = None
        self.env_trackers: Dict[str, Trackable] = {}
        self.rollout_params: Dict[str, Any] = {}
        self.load_path: Optional[str] = None
        self.rollouts = asyncio.Queue()
        self.last_rollout: Optional["Rollout"] = None

        self.evaluator: Optional[AbstractEvaluator] = None

        self.checkpoint_history_size = history_size
        self._ckpts_circular_queue: List[CheckpointState] = []
        self._latest_ckpt_idx = -1

    def register_env_tracker(self, env_tracker: Trackable) -> None:
        assert (
            env_tracker.name not in self.env_trackers
        ), f"{env_tracker.name} already registered"
        if self.load_path:
            env_tracker.load(self.load_path)
        self.env_trackers[env_tracker.name] = env_tracker

    async def get_learner_view(self) -> Optional[LearnerView]:
        if self.is_closed:
            return None
        rollouts = []
        if self.rollouts.empty():
            if self.last_rollout:
                return LearnerView(
                    rollouts=(self.last_rollout,),
                    latest_checkpoint_policy=self.latest_checkpoint_policy,
                )
            rollouts.append(await self.rollouts.get())
        while not self.rollouts.empty():
            try:
                rollouts.append(self.rollouts.get_nowait())
            except asyncio.QueueEmpty:
                break
        self.last_rollout = rollouts[-1]
        non_none_rollouts = tuple(r for r in rollouts if r is not None)
        if not non_none_rollouts:
            return None
        return LearnerView(
            rollouts=non_none_rollouts,
            latest_checkpoint_policy=self.latest_checkpoint_policy,
        )

    def initialize_learner(
        self, learner_initialize_data: RemoteLearnerInitializeData
    ) -> None:
        (self.latest_policy, algo_state, load_path) = learner_initialize_data
        if load_path:
            self.load(load_path)
        if self.checkpoint_history_size:
            self.submit_checkpoint(self._generate_checkpoint_state(algo_state))

    def submit_learner_update(self, learner_update: RemoteLearnerUpdate) -> None:
        (
            self.latest_policy,
            self.rollout_params,
            self.timesteps_elapsed,
            eval_enqueue,
        ) = learner_update

        if eval_enqueue:
            self.enqueue_latest_policy(eval_enqueue)

    def update_for_rollout_start(self) -> Optional[RolloutView]:
        if self.is_closed:
            return None
        assert self.latest_policy is not None, f"Must initialize_learner first"
        return RolloutView(
            self.latest_policy,
            env_state={k: v.get_state() for k, v in self.env_trackers.items()},
            checkpoint_policies=tuple(
                ckpt.policy for ckpt in self._ckpts_circular_queue
            ),
            latest_checkpoint_idx=self._latest_ckpt_idx,
            rollout_params=self.rollout_params,
            timesteps_elapsed=self.timesteps_elapsed,
        )

    def submit_rollout_update(self, rollout_update: RolloutUpdate) -> None:
        (rollout, env_state) = rollout_update
        self.rollouts.put_nowait(rollout)
        for k, tracker in self.env_trackers.items():
            tracker.set_state(env_state[k])

    def submit_checkpoint(self, checkpoint: CheckpointState) -> None:
        assert (
            self.checkpoint_history_size > 0
        ), "Shouldn't have submitted checkpoint given no checkpoint history"
        if len(self._ckpts_circular_queue) < self.checkpoint_history_size:
            self._latest_ckpt_idx = len(self._ckpts_circular_queue)
            self._ckpts_circular_queue.append(checkpoint)
        else:
            self._latest_ckpt_idx = (
                self._latest_ckpt_idx + 1
            ) % self.checkpoint_history_size
            self._ckpts_circular_queue[self._latest_ckpt_idx] = checkpoint

    def load(self, path: str) -> None:
        self.load_path = path
        if self.latest_policy is not None:
            self.latest_policy.load(path)
        for tracker in self.env_trackers.values():
            tracker.load(path)

    def initialize_evaluator(self, evaluator: AbstractEvaluator) -> None:
        self.evaluator = evaluator

    def enqueue_latest_policy(
        self,
        eval_enqueue: RemoteEvalEnqueue,
    ) -> None:
        assert self.evaluator is not None, "evaluator not initialized"
        self.evaluator.enqueue_eval(
            self._generate_eval_view(eval_enqueue.algo_state),
        )

    def evaluate_latest_policy(
        self,
        eval_enqueue: RemoteEvalEnqueue,
        n_episodes: Optional[int] = None,
        print_returns: bool = False,
    ) -> EpisodesStats:
        assert self.evaluator is not None, "evaluator not initialized"
        return self.evaluator.evaluate(
            self._generate_eval_view(eval_enqueue.algo_state),
            n_episodes=n_episodes,
            print_returns=print_returns,
        )

    def close(self) -> DataStoreFinalization:
        self.is_closed = True
        # None is a sentinel value to stop waiting for rollouts
        self.rollouts.put_nowait(None)
        assert self.evaluator is not None, "evaluator not initialized"
        return DataStoreFinalization(best_eval_stats=self.evaluator.best_eval_stats)

    @property
    def latest_checkpoint(self) -> Optional[CheckpointState]:
        return (
            self._ckpts_circular_queue[self._latest_ckpt_idx]
            if self._ckpts_circular_queue
            else None
        )

    @property
    def latest_checkpoint_policy(self) -> Optional["Policy"]:
        latest_checkpoint = self.latest_checkpoint
        if latest_checkpoint is None:
            return None
        return latest_checkpoint.policy

    def _generate_checkpoint_state(self, algo_state: Trackable) -> CheckpointState:
        assert self.latest_policy is not None, "Must initialize_learner first"
        return CheckpointState(
            self.latest_policy,
            algo_state,
            {k: v.get_state() for k, v in self.env_trackers.items()},
        )

    def _generate_eval_view(self, algo_state: Trackable) -> EvalView:
        return EvalView(
            *self._generate_checkpoint_state(algo_state),
            checkpoint_policies=tuple(
                ckpt.policy for ckpt in self._ckpts_circular_queue
            ),
            latest_checkpoint_idx=self._latest_ckpt_idx,
            timesteps_elapsed=self.timesteps_elapsed,
        )
