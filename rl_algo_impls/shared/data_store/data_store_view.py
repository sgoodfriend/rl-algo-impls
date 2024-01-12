from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional

from rl_algo_impls.rollout.rollout import Rollout
from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.data_store.checkpoint_policies_delegate import (
    CheckpointPoliciesDelegate,
)
from rl_algo_impls.shared.data_store.data_store_data import (
    CheckpointState,
    EvalDataStoreViewView,
    EvalEnqueue,
    EvalView,
    LearnerDataStoreViewUpdate,
    LearnerUpdate,
    LearnerView,
    RolloutDataStoreViewView,
    RolloutUpdate,
)
from rl_algo_impls.shared.data_store.in_process_data_store_accessor import (
    InProcessDataStoreAccessor,
)
from rl_algo_impls.shared.evaluator.abstract_evaluator import AbstractEvaluator
from rl_algo_impls.shared.policy.abstract_policy import AbstractPolicy
from rl_algo_impls.shared.trackable import UpdateTrackable


class DataStoreView:
    def __init__(self, data_store_accessor: AbstractDataStoreAccessor):
        self.data_store_accessor = data_store_accessor


class LearnerDataStoreView(DataStoreView):
    def __init__(self, data_store_accessor: AbstractDataStoreAccessor):
        super().__init__(data_store_accessor)
        self.rollout_params: Dict[str, Any] = {}

        self.evaluator: Optional[AbstractEvaluator] = None
        self.num_evaluations = 0

    def get_learner_view(self, wait: bool = False) -> LearnerView:
        return self.data_store_accessor.get_learner_view(wait=wait)

    def submit_learner_update(self, update: LearnerDataStoreViewUpdate) -> None:
        assert self.evaluator is not None, "evaluator not initialized"
        if self.evaluator.should_eval_on_timesteps_elapsed(
            update.timesteps_elapsed, self.num_evaluations
        ):
            eval_enqueue = EvalEnqueue(update.algo)
            self.num_evaluations += 1
        else:
            eval_enqueue = None
        self.data_store_accessor.submit_learner_update(
            LearnerUpdate(
                policy=update.policy,
                rollout_params=self.rollout_params,
                timesteps_elapsed=update.timesteps_elapsed,
                eval_enqueue=eval_enqueue,
            )
        )
        self.rollout_params = {}

    def update_rollout_param(self, key: str, value: Any) -> None:
        self.rollout_params[key] = value

    def initialize_evaluator(self, evaluator: AbstractEvaluator) -> None:
        self.evaluator = evaluator
        self.data_store_accessor.initialize_evaluator(evaluator)


class VectorEnvDataStoreView(DataStoreView):
    def __init__(self, data_store_accessor: AbstractDataStoreAccessor):
        super().__init__(data_store_accessor)
        self.env_trackers: DefaultDict[str, List[UpdateTrackable]] = defaultdict(list)
        self.checkpoint_policy_trackers = []

    def add_trackable(self, trackable: UpdateTrackable) -> None:
        self.env_trackers[trackable.name].append(trackable)

    def add_checkpoint_policy_delegate(
        self, checkpoint_policy_delegate: CheckpointPoliciesDelegate
    ) -> None:
        assert isinstance(checkpoint_policy_delegate, CheckpointPoliciesDelegate)
        self.checkpoint_policy_trackers.append(checkpoint_policy_delegate)


class RolloutDataStoreView(VectorEnvDataStoreView):
    def update_for_rollout_start(self) -> Optional[RolloutDataStoreViewView]:
        rollout_view = self.data_store_accessor.update_for_rollout_start()
        if rollout_view is None:
            return None
        (
            policy,
            env_state,
            checkpoint_policies,
            latest_checkpoint_idx,
            rollout_params,
            timesteps_elapsed,
        ) = rollout_view
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
                env_update={
                    k: self.env_trackers[k][0].get_update() for k in self.env_trackers
                },
            )
        )

    def add_trackable(self, trackable: UpdateTrackable) -> None:
        super().add_trackable(trackable)
        assert len(self.env_trackers[trackable.name]) == 1
        self.data_store_accessor.register_env_tracker(trackable)


class EvalDataStoreView(VectorEnvDataStoreView):
    def __init__(
        self, data_store_accessor: AbstractDataStoreAccessor, is_eval_job: bool = False
    ):
        super().__init__(data_store_accessor)
        self.is_eval_job = is_eval_job

    def update_from_eval_data(self, eval_data: EvalView) -> EvalDataStoreViewView:
        (
            policy,
            self.algo_state,
            self.env_state,
            checkpoint_policies,
            latest_checkpoint_idx,
            timesteps_elapsed,
        ) = eval_data
        for k, trackers in self.env_trackers.items():
            for t in trackers:
                t.set_state(self.env_state[k])
        for cpt in self.checkpoint_policy_trackers:
            cpt.update_checkpoint_policies(checkpoint_policies, latest_checkpoint_idx)
        return EvalDataStoreViewView(policy, timesteps_elapsed)

    def submit_checkpoint(self, policy: AbstractPolicy) -> None:
        self.data_store_accessor.submit_checkpoint(
            CheckpointState(
                policy=policy,
                algo_state=self.algo_state,
                env_state=self.env_state,
            )
        )

    def save(self, policy: AbstractPolicy, save_path: str) -> None:
        policy.save(save_path)
        self.algo_state.save(save_path)
        for state in self.env_state.values():
            state.save(save_path)

    def add_trackable(self, trackable: UpdateTrackable) -> None:
        super().add_trackable(trackable)
        if self.is_eval_job:
            self.data_store_accessor.register_env_tracker(trackable)
        elif isinstance(self.data_store_accessor, InProcessDataStoreAccessor):
            trackable.set_state(
                self.data_store_accessor._data_store.env_trackers[
                    trackable.name
                ].get_state()
            )
