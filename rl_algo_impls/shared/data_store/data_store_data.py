from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Tuple

from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.trackable import Trackable

if TYPE_CHECKING:
    from rl_algo_impls.rollout.rollout import Rollout
    from rl_algo_impls.shared.algorithm import Algorithm
    from rl_algo_impls.shared.policy.policy import Policy


class LearnerView(NamedTuple):
    rollouts: Tuple["Rollout", ...]
    latest_checkpoint_policy: Optional["Policy"]


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


class RolloutView(NamedTuple):
    policy: "Policy"
    env_state: Dict[str, Any]
    checkpoint_policies: Tuple["Policy", ...]
    latest_checkpoint_idx: int
    rollout_params: Dict[str, Any]
    timesteps_elapsed: int


class RolloutUpdate(NamedTuple):
    rollout: "Rollout"
    env_update: Dict[str, Any]


class EvalView(NamedTuple):
    policy: "Policy"
    algo_state: Trackable
    env_state: Dict[str, Any]
    checkpoint_policies: Tuple["Policy", ...]
    latest_checkpoint_idx: int
    timesteps_elapsed: int


class LearnerDataStoreViewUpdate(NamedTuple):
    policy: "Policy"
    algo: "Algorithm"
    timesteps_elapsed: int


class RolloutDataStoreViewView(NamedTuple):
    policy: "Policy"
    rollout_params: Dict[str, Any]
    timesteps_elapsed: int


class EvalDataStoreViewView(NamedTuple):
    policy: "Policy"
    timesteps_elapsed: int


class CheckpointState(NamedTuple):
    policy: "Policy"
    algo_state: Trackable
    env_state: Dict[str, Any]


class DataStoreFinalization(NamedTuple):
    best_eval_stats: Optional[EpisodesStats] = None
