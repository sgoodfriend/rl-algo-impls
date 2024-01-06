from abc import ABC, abstractmethod
from typing import Optional

from rl_algo_impls.shared.data_store.data_store_data import (
    CheckpointState,
    DataStoreFinalization,
    EvalEnqueue,
    LearnerInitializeData,
    LearnerUpdate,
    LearnerView,
    RolloutUpdate,
    RolloutView,
)
from rl_algo_impls.shared.evaluator.abstract_evaluator import AbstractEvaluator
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.trackable import UpdateTrackable


class AbstractDataStoreAccessor(ABC):
    @abstractmethod
    def register_env_tracker(self, env_tracker: UpdateTrackable) -> None:
        ...

    @abstractmethod
    def get_learner_view(self, wait: bool = False) -> LearnerView:
        ...

    @abstractmethod
    def initialize_learner(
        self, learner_initialize_data: LearnerInitializeData
    ) -> None:
        ...

    @abstractmethod
    def submit_learner_update(self, learner_update: LearnerUpdate) -> None:
        ...

    @abstractmethod
    def update_for_rollout_start(self) -> Optional[RolloutView]:
        ...

    @abstractmethod
    def submit_rollout_update(self, rollout_update: RolloutUpdate) -> None:
        ...

    @abstractmethod
    def submit_checkpoint(self, checkpoint: CheckpointState) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...

    @abstractmethod
    def initialize_evaluator(self, evaluator: AbstractEvaluator) -> None:
        ...

    @abstractmethod
    def evaluate_latest_policy(
        self,
        eval_enqueue: EvalEnqueue,
        n_episodes: Optional[int] = None,
        print_returns: bool = False,
    ) -> EpisodesStats:
        ...

    @abstractmethod
    def close(self) -> DataStoreFinalization:
        ...
