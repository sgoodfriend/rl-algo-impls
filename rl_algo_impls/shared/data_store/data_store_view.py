from abc import abstractmethod
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

from rl_algo_impls.rollout.rollout import Rollout
from rl_algo_impls.shared.data_store.checkpoint_policies_delegate import (
    CheckpointPoliciesDelegate,
)
from rl_algo_impls.shared.data_store.data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.data_store.data_store_data import (
    CheckpointState,
    EvalDataStoreViewView,
    LearnerDataStoreViewUpdate,
    LearnerUpdate,
    LearnerView,
    RolloutDataStoreViewView,
    RolloutUpdate,
)
from rl_algo_impls.shared.data_store.synchronous_data_store_accessor import (
    SynchronousDataStoreAccessor,
)
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.trackable import Trackable


class DataStoreView:
    def __init__(self, data_store_accessor: AbstractDataStoreAccessor):
        self.data_store_accessor = data_store_accessor


class LearnerDataStoreView(DataStoreView):
    def __init__(self, data_store_accessor: AbstractDataStoreAccessor):
        super().__init__(data_store_accessor)
        self.rollout_params: Dict[str, Any] = {}

    def get_learner_view(self) -> LearnerView:
        return self.data_store_accessor.get_learner_view()

    def submit_learner_update(self, update: LearnerDataStoreViewUpdate) -> None:
        self.data_store_accessor.submit_learner_update(
            LearnerUpdate(**update._asdict(), rollout_params=self.rollout_params)
        )
        self.rollout_params = {}

    def update_rollout_param(self, key: str, value: Any) -> None:
        self.rollout_params[key] = value


class VectorEnvDataStoreView(DataStoreView):
    def __init__(self, data_store_accessor: AbstractDataStoreAccessor):
        super().__init__(data_store_accessor)
        self.env_trackers: DefaultDict[str, List[Trackable]] = defaultdict(list)
        self.checkpoint_policy_trackers = []

    @abstractmethod
    def add_trackable(self, trackable: Trackable) -> None:
        self.env_trackers[trackable.name].append(trackable)

    @abstractmethod
    def add_checkpoint_policy_delegate(
        self, checkpoint_policy_delegate: CheckpointPoliciesDelegate
    ) -> None:
        assert isinstance(checkpoint_policy_delegate, CheckpointPoliciesDelegate)
        self.checkpoint_policy_trackers.append(checkpoint_policy_delegate)


class RolloutDataStoreView(VectorEnvDataStoreView):
    def update_for_rollout_start(self) -> RolloutDataStoreViewView:
        (
            policy,
            env_state,
            checkpoint_policies,
            latest_checkpoint_idx,
            rollout_params,
            timesteps_elapsed,
        ) = self.data_store_accessor.update_for_rollout_start()
        assert set(self.env_trackers) == set(env_state)
        for k, trackers in self.env_trackers.items():
            assert len(trackers) == 1
            trackers[0].set_state(env_state[k])
        for cpt in self.checkpoint_policy_trackers:
            cpt.update_checkpoint_policies(checkpoint_policies, latest_checkpoint_idx)
        return RolloutDataStoreViewView(policy, rollout_params, timesteps_elapsed)

    def submit_rollout_update(self, rollout: Rollout) -> None:
        self.data_store_accessor.submit_rollout_update(
            RolloutUpdate(
                rollout=rollout,
                env_state={
                    k: self.env_trackers[k][0].get_state() for k in self.env_trackers
                },
            )
        )

    def add_trackable(self, trackable: Trackable) -> None:
        super().add_trackable(trackable)
        assert len(self.env_trackers[trackable.name]) == 1
        self.data_store_accessor.register_env_tracker(trackable)


class EvalDataStoreView(VectorEnvDataStoreView):
    def __init__(
        self, data_store_accessor: AbstractDataStoreAccessor, is_eval_job: bool = False
    ):
        super().__init__(data_store_accessor)
        self.is_eval_job = is_eval_job

    def update_for_eval_start(self) -> EvalDataStoreViewView:
        (
            policy,
            self.algo_state,
            self.env_state,
            checkpoint_policies,
            latest_checkpoint_idx,
            timesteps_elapsed,
        ) = self.data_store_accessor.update_for_eval_start()
        for k, trackers in self.env_trackers.items():
            for t in trackers:
                t.set_state(self.env_state[k])
        for cpt in self.checkpoint_policy_trackers:
            cpt.update_checkpoint_policies(checkpoint_policies, latest_checkpoint_idx)
        return EvalDataStoreViewView(policy, timesteps_elapsed)

    def submit_checkpoint(self, policy: Policy) -> None:
        self.data_store_accessor.submit_checkpoint(
            CheckpointState(
                policy=policy,
                algo_state=self.algo_state,
                env_state=self.env_state,
            )
        )

    def save(self, policy: Policy, save_path: str) -> None:
        policy.save(save_path)
        self.algo_state.save(save_path)
        for trackers in self.env_trackers.values():
            if trackers:
                trackers[0].save(save_path)

    def add_trackable(self, trackable: Trackable) -> None:
        super().add_trackable(trackable)
        if self.is_eval_job:
            self.data_store_accessor.register_env_tracker(trackable)
        elif isinstance(self.data_store_accessor, SynchronousDataStoreAccessor):
            trackable.set_state(
                self.data_store_accessor._data_store.env_trackers[
                    trackable.name
                ].get_state()
            )
