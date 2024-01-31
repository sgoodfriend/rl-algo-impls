import logging
from typing import TYPE_CHECKING, List, Optional

import ray

if TYPE_CHECKING:
    from rl_algo_impls.runner.config import Config
    from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
        AbstractDataStoreAccessor,
    )
    from rl_algo_impls.shared.data_store.data_store_data import EvalView
    from rl_algo_impls.shared.policy.policy import Policy
    from rl_algo_impls.shared.stats import EpisodesStats
    from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
        AbstractSummaryWrapper,
    )


@ray.remote
class EvaluatorActor:
    def __init__(
        self,
        config: "Config",
        data_store_accessor: "AbstractDataStoreAccessor",
        tb_writer: "AbstractSummaryWrapper",
        **kwargs,
    ) -> None:
        from rl_algo_impls.utils.ray import init_ray_actor

        evaluator_cuda_index = config.evaluator_cuda_index
        init_ray_actor(
            cuda_visible_devices=[evaluator_cuda_index]
            if evaluator_cuda_index is not None
            else []
        )
        logging.basicConfig(level=logging.INFO, handlers=[])
        tb_writer.maybe_add_logging_handler()

        from rl_algo_impls.shared.evaluator.evaluator import Evaluator

        self.evaluator = Evaluator(config, data_store_accessor, tb_writer, **kwargs)

    def best_eval_stats(self) -> Optional["EpisodesStats"]:
        return self.evaluator.best

    def evaluate(
        self,
        eval_data: "EvalView",
        n_episodes: Optional[int] = None,
        print_returns: bool = False,
    ) -> "EpisodesStats":
        return self.evaluator.evaluate(eval_data, n_episodes, print_returns)

    def save(self, policy: "Policy", model_path: str) -> None:
        self.evaluator.save(policy, model_path)
