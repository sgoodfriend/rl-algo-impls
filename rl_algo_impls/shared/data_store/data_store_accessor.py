from abc import ABC, abstractmethod

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


class AbstractDataStoreAccessor(ABC):
    @abstractmethod
    def register_env_tracker(self, env_tracker: Trackable) -> None:
        ...

    @abstractmethod
    def get_learner_view(self) -> LearnerView:
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
    def update_for_rollout_start(self) -> RolloutView:
        ...

    @abstractmethod
    def submit_rollout_update(self, rollout_update: RolloutUpdate) -> None:
        ...

    @abstractmethod
    def update_for_eval_start(self) -> EvalView:
        ...

    @abstractmethod
    def submit_checkpoint(self, checkpoint: CheckpointState) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...
