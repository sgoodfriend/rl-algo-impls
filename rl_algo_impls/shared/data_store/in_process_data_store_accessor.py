from typing import TYPE_CHECKING, Optional

from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.data_store.data_store_data import (
    CheckpointState,
    DataStoreFinalization,
    EvalEnqueue,
    EvalView,
    LearnerInitializeData,
    LearnerUpdate,
    LearnerView,
    RolloutUpdate,
    RolloutView,
)
from rl_algo_impls.shared.data_store.in_process_data_store import InProcessDataStore
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.trackable import UpdateTrackable

if TYPE_CHECKING:
    from rl_algo_impls.rollout.in_process_rollout_generator import (
        InProcessRolloutGenerator,
    )
    from rl_algo_impls.shared.algorithm import Algorithm
    from rl_algo_impls.shared.evaluator.in_process_evaluator import InProcessEvaluator


class InProcessDataStoreAccessor(AbstractDataStoreAccessor):
    def __init__(self, history_size: int = 0):
        self._data_store = InProcessDataStore(history_size)
        self.rollout_generator: Optional["InProcessRolloutGenerator"] = None
        self.evaluator: Optional["InProcessEvaluator"] = None

    def register_env_tracker(self, env_tracker: UpdateTrackable) -> None:
        self._data_store.env_trackers[env_tracker.name] = env_tracker

    def get_learner_view(self, wait: bool = False) -> LearnerView:
        assert self.rollout_generator is not None
        rollouts = [self.rollout_generator.rollout()]
        latest_checkpoint = self._data_store.latest_checkpoint
        return LearnerView(
            rollouts=tuple(rollouts),
            latest_checkpoint_policy=latest_checkpoint.policy
            if latest_checkpoint
            else None,
        )

    def initialize_learner(
        self, learner_initialize_data: LearnerInitializeData
    ) -> None:
        (
            self._data_store.policy,
            algo,
            load_path,
        ) = learner_initialize_data
        if load_path:
            self._data_store.load(load_path)
            algo.load(load_path)
        if self._data_store.checkpoint_history_size:
            self.submit_checkpoint(self._generate_checkpoint_state(algo))

    def submit_learner_update(self, learner_update: LearnerUpdate) -> None:
        (
            self._data_store.policy,
            self._data_store.rollout_params,
            self._data_store.timesteps_elapsed,
            eval_enqueue,
        ) = learner_update
        if eval_enqueue is not None:
            self.evaluate_latest_policy(eval_enqueue)

    def update_for_rollout_start(self) -> RolloutView:
        return RolloutView(
            self._data_store.policy,
            env_state={
                k: v.get_state() for k, v in self._data_store.env_trackers.items()
            },
            checkpoint_policies=tuple(
                ckpt.policy for ckpt in self._data_store.checkpoints
            ),
            latest_checkpoint_idx=self._data_store._latest_ckpt_idx,
            rollout_params=self._data_store.rollout_params,
            timesteps_elapsed=self._data_store.timesteps_elapsed,
        )

    def submit_rollout_update(self, rollout_update: RolloutUpdate) -> None:
        # rollout ignored since rollout returns in get_learner_view
        # env_update ignored since _data_store is using the rollout env_trackers
        pass

    def submit_checkpoint(self, checkpoint: CheckpointState) -> None:
        self._data_store.create_checkpoint(checkpoint)

    def load(self, path: str) -> None:
        self._data_store.load(path)

    def initialize_evaluator(
        self,
        evaluator: "InProcessEvaluator",
    ) -> None:
        self.evaluator = evaluator

    def enqueue_latest_policy(self, eval_enqueue: EvalEnqueue) -> None:
        assert self.evaluator is not None, "evaluator not initialized"
        self.evaluator.enqueue_eval(
            self._generate_eval_view(eval_enqueue.algo),
        )

    def evaluate_latest_policy(
        self,
        eval_enqueue: EvalEnqueue,
        n_episodes: Optional[int] = None,
        print_returns: bool = False,
    ) -> EpisodesStats:
        assert self.evaluator is not None, "evaluator not initialized"
        return self.evaluator.evaluate(
            self._generate_eval_view(eval_enqueue.algo),
            n_episodes,
            print_returns,
        )

    def close(self) -> DataStoreFinalization:
        return DataStoreFinalization(best_eval_stats=self.evaluator.best_eval_stats)

    def _generate_checkpoint_state(self, algo: "Algorithm") -> CheckpointState:
        from rl_algo_impls.shared.data_store.algorithm_state import (
            SynchronousAlgorithmState,
        )

        return CheckpointState(
            policy=self._data_store.policy,
            algo_state=SynchronousAlgorithmState(algo),
            env_state={
                k: v.get_state() for k, v in self._data_store.env_trackers.items()
            },
        )

    def _generate_eval_view(self, algo: "Algorithm") -> EvalView:
        return EvalView(
            *self._generate_checkpoint_state(algo),
            checkpoint_policies=tuple(
                ckpt.policy for ckpt in self._data_store.checkpoints
            ),
            latest_checkpoint_idx=self._data_store._latest_ckpt_idx,
            timesteps_elapsed=self._data_store.timesteps_elapsed,
        )
