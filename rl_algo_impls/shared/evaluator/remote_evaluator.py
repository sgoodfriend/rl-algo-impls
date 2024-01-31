from typing import List, Optional, Union

import ray

from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.data_store.data_store_data import EvalEnqueue, EvalView
from rl_algo_impls.shared.evaluator.abstract_evaluator import AbstractEvaluator
from rl_algo_impls.shared.evaluator.evaluator_actor import EvaluatorActor
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper


class RemoteEvaluator(AbstractEvaluator):
    def __init__(
        self,
        config: Config,
        data_store_accessor: AbstractDataStoreAccessor,
        tb_writer: AbstractSummaryWrapper,
        self_play_wrapper: Optional[SelfPlayWrapper] = None,
        step_freq: Union[int, float] = 50_000,
        skip_evaluate_at_start: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(int(step_freq), skip_evaluate_at_start)
        assert (
            not self_play_wrapper
        ), "SelfPlayWrapper not supported for RemoteEvaluator"
        self.data_store_accessor = data_store_accessor
        self.eval_actor = EvaluatorActor.remote(
            config, data_store_accessor, tb_writer, **kwargs
        )

    @property
    def best_eval_stats(self) -> Optional[EpisodesStats]:
        return ray.get(self.eval_actor.best_eval_stats.remote())

    def enqueue_eval(self, eval_data: EvalView) -> None:
        self.eval_actor.evaluate.remote(eval_data)

    def evaluate(
        self,
        eval_data: EvalView,
        n_episodes: Optional[int] = None,
        print_returns: bool = False,
    ) -> EpisodesStats:
        return ray.get(
            self.eval_actor.evaluate.remote(eval_data, n_episodes, print_returns)
        )

    def evaluate_latest_policy(
        self,
        algorithm: Algorithm,
        n_episodes: Optional[int] = None,
        print_returns: bool = False,
    ) -> EpisodesStats:
        return self.data_store_accessor.evaluate_latest_policy(
            EvalEnqueue(algorithm), n_episodes, print_returns
        )

    def save(self, policy: Algorithm, model_path: str) -> None:
        ray.get(self.eval_actor.save.remote(policy, model_path))
