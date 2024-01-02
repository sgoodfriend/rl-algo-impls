from typing import List, Optional, Union

from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.data_store.data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.data_store.synchronous_data_store_accessor import (
    SynchronousDataStoreAccessor,
)
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
        self_play_wrapper: Optional[SelfPlayWrapper] = None,
        best_model_path: Optional[str] = None,
        step_freq: Union[int, float] = 50_000,
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
        skip_evaluate_at_start: bool = False,
        only_checkpoint_best_policies: bool = False,
        latest_model_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert isinstance(
            data_store_accessor, SynchronousDataStoreAccessor
        ), f"Only SynchronousDataStoreAccessor supported, not {data_store_accessor.__class__.__name__}"
        data_store_accessor.evaluator = self
        self.evaluator = Evaluator(
            config,
            data_store_accessor,
            tb_writer,
            self_play_wrapper=self_play_wrapper,
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
        self.skip_evaluate_at_start = skip_evaluate_at_start
        self.step_freq = int(step_freq)

        self.timesteps_elapsed = 0
        self.num_evaluations = 0

    def on_timesteps_elapsed(self, timesteps_elapsed: int) -> bool:
        self.timesteps_elapsed = timesteps_elapsed
        desired_num_evaluations = self.timesteps_elapsed // self.step_freq
        if not self.skip_evaluate_at_start:
            desired_num_evaluations += 1
        if desired_num_evaluations > self.num_evaluations:
            self.evaluate()
            self.num_evaluations += 1
        return True

    def close(self) -> Optional[EpisodesStats]:
        return self.evaluator.best

    def evaluate(
        self, n_episodes: Optional[int] = None, print_returns: Optional[bool] = None
    ) -> EpisodesStats:
        return self.evaluator.evaluate(n_episodes, print_returns)

    def generate_video(self, policy: Policy) -> None:
        return self.evaluator.generate_video(policy)

    def save(self, policy: Policy, model_path: str) -> None:
        self.evaluator.save(policy, model_path)
