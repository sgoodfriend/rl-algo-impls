from typing import List, Optional, Union

from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.data_store.data_store_data import EvalEnqueue, EvalView
from rl_algo_impls.shared.evaluator.abstract_evaluator import AbstractEvaluator
from rl_algo_impls.shared.evaluator.evaluator import Evaluator
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper


class InProcessEvaluator(AbstractEvaluator):
    def __init__(
        self,
        config: Config,
        data_store_accessor: AbstractDataStoreAccessor,
        tb_writer: AbstractSummaryWrapper,
        step_freq: Union[int, float] = 50_000,
        skip_evaluate_at_start: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(int(step_freq), skip_evaluate_at_start)
        self.data_store_accessor = data_store_accessor
        self.evaluator = Evaluator(
            config,
            data_store_accessor,
            tb_writer,
            **kwargs,
        )

    @property
    def best_eval_stats(self) -> Optional[EpisodesStats]:
        return self.evaluator.best

    def enqueue_eval(self, eval_data: EvalView) -> None:
        self.evaluator.evaluate(eval_data)

    def evaluate(
        self,
        eval_data: EvalView,
        n_episodes: Optional[int] = None,
        print_returns: bool = False,
    ) -> EpisodesStats:
        return self.evaluator.evaluate(eval_data, n_episodes, print_returns)

    def evaluate_latest_policy(
        self,
        algorithm: Algorithm,
        n_episodes: Optional[int] = None,
        print_returns: bool = False,
    ) -> EpisodesStats:
        return self.data_store_accessor.evaluate_latest_policy(
            EvalEnqueue(algorithm), n_episodes, print_returns
        )

    def save(self, policy: Policy, model_path: str) -> None:
        self.evaluator.save(policy, model_path)
