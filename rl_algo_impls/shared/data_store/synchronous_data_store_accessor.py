from typing import TYPE_CHECKING, Optional

from rl_algo_impls.shared.data_store.synchronous_data_store import SynchronousDataStore
from rl_algo_impls.shared.data_store.data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.data_store.data_store_data import (
    CheckpointState,
    EvalView,
    LearnerInitializeData,
    LearnerUpdate,
    LearnerView,
    RolloutUpdate,
    RolloutView,
)
from rl_algo_impls.shared.trackable import Trackable

if TYPE_CHECKING:
    from rl_algo_impls.rollout.in_process_rollout import InProcessRolloutGenerator
    from rl_algo_impls.shared.data_store.evaluator import Evaluator


class SynchronousDataStoreAccessor(AbstractDataStoreAccessor):
    def __init__(self, history_size: int = 0):
        self._data_store = SynchronousDataStore(history_size)
        self.rollout_generator: Optional["InProcessRolloutGenerator"] = None
        self.evaluator: Optional["Evaluator"] = None

    def register_env_tracker(self, env_tracker: Trackable) -> None:
        self._data_store.env_trackers[env_tracker.name] = env_tracker

    def get_learner_view(self) -> LearnerView:
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
            self._data_store.algo,
        ) = learner_initialize_data

    def submit_learner_update(self, learner_update: LearnerUpdate) -> None:
        (
            self._data_store.policy,
            self._data_store.algo,
            self._data_store.rollout_params,
            self._data_store.timesteps_elapsed,
        ) = learner_update
        assert self.evaluator is not None
        self.evaluator.on_timesteps_elapsed(self._data_store.timesteps_elapsed)

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

    def update_for_eval_start(self) -> EvalView:
        from rl_algo_impls.shared.data_store.algorithm_state import (
            SynchronousAlgorithmState,
        )

        return EvalView(
            policy=self._data_store.policy,
            algo_state=SynchronousAlgorithmState(self._data_store.algo),
            env_state={
                k: v.get_state() for k, v in self._data_store.env_trackers.items()
            },
            checkpoint_policies=tuple(
                ckpt.policy for ckpt in self._data_store.checkpoints
            ),
            latest_checkpoint_idx=self._data_store._latest_ckpt_idx,
            timesteps_elapsed=self._data_store.timesteps_elapsed,
        )

    def submit_checkpoint(self, checkpoint: CheckpointState) -> None:
        self._data_store.create_checkpoint(checkpoint)

    def load(self, path: str) -> None:
        self._data_store.load(path)
