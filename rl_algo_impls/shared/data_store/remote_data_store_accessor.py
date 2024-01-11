from copy import deepcopy
from typing import Optional

import ray
import torch

from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.data_store.algorithm_state import RemoteAlgorithmState
from rl_algo_impls.shared.data_store.data_store_actor import (
    DataStoreActor,
    RemoteEvalEnqueue,
    RemoteLearnerInitializeData,
    RemoteLearnerUpdate,
)
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
from rl_algo_impls.shared.policy.policy_state import RemotePolicyState
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.trackable import UpdateTrackable


class RemoteDataStoreAccessor(AbstractDataStoreAccessor):
    def __init__(self, history_size: int = 0) -> None:
        self.checkpoint_history_size = history_size
        self.data_store_actor = DataStoreActor.remote(history_size)

    def register_env_tracker(self, env_tracker: UpdateTrackable) -> None:
        self.data_store_actor.register_env_tracker.remote(env_tracker)

    def get_learner_view(self, wait: bool = False) -> LearnerView:
        return ray.get(self.data_store_actor.get_learner_view.remote(wait=wait))

    def initialize_learner(
        self, learner_initialize_data: LearnerInitializeData
    ) -> None:
        policy, algo, load_path = learner_initialize_data
        if load_path:
            policy.load(load_path)
            algo.load(load_path)
        ray.get(
            self.data_store_actor.initialize_learner.remote(
                RemoteLearnerInitializeData(
                    policy=deepcopy(policy).to(torch.device("cpu")),
                    algo_state=RemoteAlgorithmState(algo),
                    load_path=load_path,
                )
            )
        )

    def submit_learner_update(self, learner_update: LearnerUpdate) -> None:
        policy, rollout_params, timesteps_elapsed, eval_enqueue = learner_update
        self.data_store_actor.submit_learner_update.remote(
            RemoteLearnerUpdate(
                RemotePolicyState(policy),
                rollout_params,
                timesteps_elapsed,
                RemoteEvalEnqueue.from_eval_enqueue(eval_enqueue),
            )
        )

    def update_for_rollout_start(self) -> Optional[RolloutView]:
        return ray.get(self.data_store_actor.update_for_rollout_start.remote())

    def submit_rollout_update(self, rollout_update: RolloutUpdate) -> None:
        self.data_store_actor.submit_rollout_update.remote(rollout_update)

    def submit_checkpoint(self, checkpoint: CheckpointState) -> None:
        if not self.checkpoint_history_size:
            return
        self.data_store_actor.submit_checkpoint.remote(
            checkpoint.to(torch.device("cpu"))
        )

    def load(self, load_path: str) -> None:
        ray.get(self.data_store_actor.load.remote(load_path))

    def initialize_evaluator(self, evaluator: AbstractEvaluator) -> None:
        self.data_store_actor.initialize_evaluator.remote(evaluator)

    def evaluate_latest_policy(
        self,
        eval_enqueue: EvalEnqueue,
        n_episodes: Optional[int] = None,
        print_returns: bool = False,
    ) -> EpisodesStats:
        return ray.get(
            self.data_store_actor.evaluate_latest_policy.remote(
                RemoteEvalEnqueue.from_eval_enqueue(eval_enqueue),
                n_episodes,
                print_returns,
            )
        )

    def close(self) -> DataStoreFinalization:
        return ray.get(self.data_store_actor.close.remote())
