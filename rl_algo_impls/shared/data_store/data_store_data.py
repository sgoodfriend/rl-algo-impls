from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Tuple, TypeVar

import torch

from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.trackable import Trackable, TrackableState

if TYPE_CHECKING:
    from rl_algo_impls.rollout.rollout import Rollout
    from rl_algo_impls.shared.algorithm import Algorithm
    from rl_algo_impls.shared.policy.policy import Policy


LearnerViewSelf = TypeVar("LearnerViewSelf", bound="LearnerView")


class LearnerView(NamedTuple):
    rollouts: Tuple["Rollout", ...]
    latest_checkpoint_policy: Optional["Policy"]

    def to(self: LearnerViewSelf, device: torch.device) -> LearnerViewSelf:
        return self.__class__(
            rollouts=self.rollouts,
            latest_checkpoint_policy=self.latest_checkpoint_policy.to(device)
            if self.latest_checkpoint_policy
            else None,
        )


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
    policy: "Policy"
    env_state: Dict[str, TrackableState]
    checkpoint_policies: Tuple["Policy", ...]
    latest_checkpoint_idx: int
    rollout_params: Dict[str, Any]
    timesteps_elapsed: int

    def to(self: RolloutViewSelf, device: torch.device) -> RolloutViewSelf:
        return self.__class__(
            policy=self.policy.to(device),
            env_state=self.env_state,
            checkpoint_policies=tuple(
                policy.to(device) for policy in self.checkpoint_policies
            ),
            latest_checkpoint_idx=self.latest_checkpoint_idx,
            rollout_params=self.rollout_params,
            timesteps_elapsed=self.timesteps_elapsed,
        )


class RolloutUpdate(NamedTuple):
    rollout: "Rollout"
    env_update: Dict[str, Any]


EvalViewSelf = TypeVar("EvalViewSelf", bound="EvalView")


class EvalView(NamedTuple):
    policy: "Policy"
    algo_state: TrackableState
    env_state: Dict[str, TrackableState]
    checkpoint_policies: Tuple["Policy", ...]
    latest_checkpoint_idx: int
    timesteps_elapsed: int

    def to(self: EvalViewSelf, device: torch.device) -> EvalViewSelf:
        return self.__class__(
            policy=self.policy.to(device),
            algo_state=self.algo_state,
            env_state=self.env_state,
            checkpoint_policies=tuple(
                policy.to(device) for policy in self.checkpoint_policies
            ),
            latest_checkpoint_idx=self.latest_checkpoint_idx,
            timesteps_elapsed=self.timesteps_elapsed,
        )


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


CheckpointStateSelf = TypeVar("CheckpointStateSelf", bound="CheckpointState")


class CheckpointState(NamedTuple):
    policy: "Policy"
    algo_state: TrackableState
    env_state: Dict[str, TrackableState]

    def to(self: CheckpointStateSelf, device: torch.device) -> CheckpointStateSelf:
        return self.__class__(
            policy=deepcopy(self.policy).to(device),
            algo_state=self.algo_state,
            env_state=self.env_state,
        )


class DataStoreFinalization(NamedTuple):
    best_eval_stats: Optional[EpisodesStats] = None
