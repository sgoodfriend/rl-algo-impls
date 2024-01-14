import asyncio
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Type, TypeVar

import ray

from rl_algo_impls.runner.config import Config
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
from rl_algo_impls.shared.policy.remote_inference_policy import RemoteInferencePolicy
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.trackable import TrackableState, UpdateTrackable
from rl_algo_impls.utils.ray import init_ray_actor

if TYPE_CHECKING:
    from rl_algo_impls.shared.policy.abstract_policy import AbstractPolicy


class RemoteLearnerInitializeData(NamedTuple):
    policy: "RemoteInferencePolicy"
    algo_state: TrackableState
    load_path: Optional[str]


RemoteEvalEnqueueSelf = TypeVar("RemoteEvalEnqueueSelf", bound="RemoteEvalEnqueue")


class RemoteEvalEnqueue(NamedTuple):
    algo_state: TrackableState

    @classmethod
    def from_eval_enqueue(
        cls: Type[RemoteEvalEnqueueSelf], eval_enqueue: Optional[EvalEnqueue]
    ) -> Optional[RemoteEvalEnqueueSelf]:
        return cls(RemoteAlgorithmState(eval_enqueue.algo)) if eval_enqueue else None


class RemoteLearnerUpdate(NamedTuple):
    rollout_params: Dict[str, Any]
    timesteps_elapsed: int
    eval_enqueue: Optional[RemoteEvalEnqueue]


@ray.remote
class DataStoreActor:
    def __init__(self, config: Config) -> None:
        init_ray_actor()
        self.config = config

        self.is_closed = False
        self.timesteps_elapsed = 0
        self.latest_policy: Optional["RemoteInferencePolicy"] = None
        self.env_trackers: Dict[str, UpdateTrackable] = {}
        self.rollout_params: Dict[str, Any] = {}
        self.load_path: Optional[str] = None
        self.rollouts = asyncio.Queue()

        self.evaluator: Optional[AbstractEvaluator] = None

        self._ckpts_circular_queue: List[CheckpointState] = []
        self._latest_ckpt_idx = -1

    def register_env_tracker(self, env_tracker: UpdateTrackable) -> None:
        if env_tracker.name in self.env_trackers:
            return
        if self.load_path:
            env_tracker.get_state().load(self.load_path)
        self.env_trackers[env_tracker.name] = env_tracker

    async def get_learner_view(self, wait: bool = False) -> Optional[LearnerView]:
        if self.is_closed:
            return None
        rollouts = []
        if self.rollouts.empty() and wait:
            rollouts.append(await self.rollouts.get())
        while not self.rollouts.empty():
            try:
                rollouts.append(self.rollouts.get_nowait())
            except asyncio.QueueEmpty:
                break
        if len(rollouts) == 1 and rollouts[0] == None:
            return None
        non_none_rollouts = tuple(r for r in rollouts if r is not None)
        return LearnerView(
            rollouts=non_none_rollouts,
            latest_checkpoint_policy=self.latest_checkpoint_policy
            if self.uses_teacher_policy
            else None,
        )

    @property
    def uses_teacher_policy(self) -> bool:
        return self.config.algo_hyperparams.get("teacher_kl_loss_coef", None)

    @property
    def latest_checkpoint(self) -> Optional[CheckpointState]:
        return (
            self._ckpts_circular_queue[self._latest_ckpt_idx]
            if self._ckpts_circular_queue
            else None
        )

    @property
    def latest_checkpoint_policy(self) -> Optional["AbstractPolicy"]:
        latest_checkpoint = self.latest_checkpoint
        if latest_checkpoint is None:
            return None
        return latest_checkpoint.policy

    def initialize_learner(
        self, learner_initialize_data: RemoteLearnerInitializeData
    ) -> None:
        (latest_policy, algo_state, load_path) = learner_initialize_data
        if load_path:
            self.load(load_path)
        # latest_policy has already loaded from load_path in RemoteDataStoreAccessor
        self.latest_policy = latest_policy
        if self.checkpoint_history_size:
            self.submit_checkpoint(self._generate_checkpoint_state(algo_state))

    def submit_learner_update(self, learner_update: RemoteLearnerUpdate) -> None:
        (
            self.rollout_params,
            self.timesteps_elapsed,
            eval_enqueue,
        ) = learner_update
        # latest_policy updated within RemoteDataStoreAccessor
        assert self.latest_policy is not None, "Must initialize_learner first"

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
        (rollout, env_update) = rollout_update
        self.rollouts.put_nowait(rollout)
        for k, tracker in self.env_trackers.items():
            tracker.apply_update(env_update[k])

    @property
    def checkpoint_history_size(self) -> int:
        return self.config.checkpoint_history_size

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
            # Reuse the old checkpoint policy process while destroying the new policy
            # process. This allows us to avoid accumulating too many processes while
            # guaranteeing rollout processes access to live actors.
            old_policy, _, _ = self._ckpts_circular_queue[self._latest_ckpt_idx]
            new_policy, new_algo_state, new_env_state = checkpoint
            assert isinstance(old_policy, RemoteInferencePolicy) and isinstance(
                new_policy, RemoteInferencePolicy
            ), "Checkpoint policy must be RemotePolicy"
            new_policy.transfer_policy_to(old_policy, delete_origin_policy=True)
            self._ckpts_circular_queue[self._latest_ckpt_idx] = CheckpointState(
                old_policy,
                new_algo_state,
                new_env_state,
            )

    def load(self, path: str) -> None:
        self.load_path = path
        assert (
            self.latest_policy is None
        ), f"latest_policy doesn't support loading in {self.__class__.__name__}"
        for tracker in self.env_trackers.values():
            tracker.get_state().load(path)

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

    def _generate_checkpoint_state(self, algo_state: TrackableState) -> CheckpointState:
        assert self.latest_policy is not None, "Must initialize_learner first"
        return CheckpointState(
            self.latest_policy.clone(self.config.inference_cuda_index),
            algo_state,
            {k: v.get_state() for k, v in self.env_trackers.items()},
        )

    def _generate_eval_view(self, algo_state: TrackableState) -> EvalView:
        return EvalView(
            *self._generate_checkpoint_state(algo_state),
            checkpoint_policies=tuple(
                ckpt.policy for ckpt in self._ckpts_circular_queue
            ),
            latest_checkpoint_idx=self._latest_ckpt_idx,
            timesteps_elapsed=self.timesteps_elapsed,
        )
