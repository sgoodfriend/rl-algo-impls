from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Tuple, TypeVar

from rl_algo_impls.shared.policy.abstract_policy import AbstractPolicy
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.trackable import TrackableState

if TYPE_CHECKING:
    from rl_algo_impls.rollout.rollout import Rollout
    from rl_algo_impls.shared.algorithm import Algorithm
    from rl_algo_impls.shared.policy.policy import Policy


LearnerViewSelf = TypeVar("LearnerViewSelf", bound="LearnerView")


class LearnerView(NamedTuple):
    rollouts: Tuple["Rollout", ...]


class LearnerInitializeData(NamedTuple):
    policy: "Policy"
    algo: "Algorithm"
    load_path: Optional[str]


class EvalEnqueue(NamedTuple):
    algo: "Algorithm"


class LearnerUpdate(NamedTuple):
    policy: "Policy"
    rollout_params: Dict[str, Any]
    timesteps_elapsed: int
    eval_enqueue: Optional[EvalEnqueue]


RolloutViewSelf = TypeVar("RolloutViewSelf", bound="RolloutView")


class RolloutView(NamedTuple):
    policy: "AbstractPolicy"
    env_state: Dict[str, TrackableState]
    checkpoint_policies: Tuple["AbstractPolicy", ...]
    latest_checkpoint_idx: int
    rollout_params: Dict[str, Any]
    timesteps_elapsed: int

    @property
    def latest_checkpoint_policy(self) -> Optional["AbstractPolicy"]:
        return (
            self.checkpoint_policies[self.latest_checkpoint_idx]
            if self.checkpoint_policies
            else None
        )


class RolloutUpdate(NamedTuple):
    rollout: "Rollout"
    env_update: Dict[str, Any]


EvalViewSelf = TypeVar("EvalViewSelf", bound="EvalView")


class EvalView(NamedTuple):
    policy: "AbstractPolicy"
    algo_state: TrackableState
    env_state: Dict[str, TrackableState]
    checkpoint_policies: Tuple["AbstractPolicy", ...]
    latest_checkpoint_idx: int
    timesteps_elapsed: int


class LearnerDataStoreViewUpdate(NamedTuple):
    policy: "Policy"
    algo: "Algorithm"
    timesteps_elapsed: int


class RolloutDataStoreViewView(NamedTuple):
    policy: "AbstractPolicy"
    rollout_params: Dict[str, Any]
    timesteps_elapsed: int
    latest_checkpoint_policy: Optional["AbstractPolicy"]


class EvalDataStoreViewView(NamedTuple):
    policy: "AbstractPolicy"
    timesteps_elapsed: int


CheckpointStateSelf = TypeVar("CheckpointStateSelf", bound="CheckpointState")


class CheckpointState(NamedTuple):
    policy: "AbstractPolicy"
    algo_state: TrackableState
    env_state: Dict[str, TrackableState]


class DataStoreFinalization(NamedTuple):
    best_eval_stats: Optional[EpisodesStats] = None
