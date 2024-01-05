import logging
from typing import List, Optional

import ray

from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.data_store.data_store_data import EvalView
from rl_algo_impls.shared.evaluator.evaluator import Evaluator
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.utils.ray import init_ray_actor


@ray.remote(num_cpus=2)
class EvaluatorActor:
    def __init__(
        self,
        config: Config,
        data_store_accessor: AbstractDataStoreAccessor,
        tb_writer: AbstractSummaryWrapper,
        best_model_path: Optional[str] = None,
        n_episodes: int = 10,
        save_best: bool = True,
        deterministic: bool = True,
        only_record_video_on_best: bool = True,
        video_dir: Optional[str] = None,
        max_video_length: int = 9000,
        ignore_first_episode: bool = False,
        additional_keys_to_log: Optional[List[str]] = None,
        score_function: str = "mean-std",
        wandb_enabled: bool = False,
        score_threshold: Optional[float] = None,
        only_checkpoint_best_policies: bool = False,
        latest_model_path: Optional[str] = None,
    ) -> None:
        init_ray_actor()
        logging.basicConfig(level=logging.INFO, handlers=[])
        tb_writer.maybe_add_logging_handler()
        self.evaluator = Evaluator(
            config,
            data_store_accessor,
            tb_writer,
            best_model_path=best_model_path,
            n_episodes=n_episodes,
            save_best=save_best,
            deterministic=deterministic,
            only_record_video_on_best=only_record_video_on_best,
            video_dir=video_dir,
            max_video_length=max_video_length,
            ignore_first_episode=ignore_first_episode,
            additional_keys_to_log=additional_keys_to_log,
            score_function=score_function,
            wandb_enabled=wandb_enabled,
            score_threshold=score_threshold,
            only_checkpoint_best_policies=only_checkpoint_best_policies,
            latest_model_path=latest_model_path,
        )

    def best_eval_stats(self) -> Optional[EpisodesStats]:
        return self.evaluator.best

    def evaluate(
        self,
        eval_data: EvalView,
        n_episodes: Optional[int] = None,
        print_returns: bool = False,
    ) -> EpisodesStats:
        return self.evaluator.evaluate(eval_data, n_episodes, print_returns)

    def save(self, policy: Policy, model_path: str) -> None:
        self.evaluator.save(policy, model_path)
