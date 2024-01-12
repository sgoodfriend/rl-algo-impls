from copy import deepcopy
from typing import Optional

import ray
import torch

from rl_algo_impls.runner.config import Config
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
from rl_algo_impls.shared.policy.remote_inference_policy import RemoteInferencePolicy
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.trackable import UpdateTrackable


class RemoteDataStoreAccessor(AbstractDataStoreAccessor):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.data_store_actor = DataStoreActor.remote(config)
        self.latest_policy: Optional[RemoteInferencePolicy] = None

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
        self.latest_policy = RemoteInferencePolicy(
            self.config.inference_cuda_index, policy
        )
        ray.get(
            self.data_store_actor.initialize_learner.remote(
                RemoteLearnerInitializeData(
                    policy=self.latest_policy,
                    algo_state=RemoteAlgorithmState(algo),
                    load_path=load_path,
                )
            )
        )

    def submit_learner_update(self, learner_update: LearnerUpdate) -> None:
        policy, rollout_params, timesteps_elapsed, eval_enqueue = learner_update
        assert self.latest_policy is not None, "Must initialize_learner first"
        self.latest_policy.set_state(policy.get_state())
        self.data_store_actor.submit_learner_update.remote(
            RemoteLearnerUpdate(
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
        if not self.config.checkpoint_history_size:
            return
        self.data_store_actor.submit_checkpoint.remote(checkpoint)

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
