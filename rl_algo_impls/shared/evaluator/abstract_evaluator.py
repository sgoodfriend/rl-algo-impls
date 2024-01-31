from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from rl_algo_impls.shared.data_store.data_store_data import EvalView
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.stats import EpisodesStats

if TYPE_CHECKING:
    from rl_algo_impls.shared.algorithm import Algorithm


class AbstractEvaluator(ABC):
    def __init__(self, step_freq: int, skip_evaluate_at_start: bool):
        self.step_freq = step_freq
        self.skip_evaluate_at_start = skip_evaluate_at_start

    def should_eval_on_timesteps_elapsed(
        self, timesteps_elapsed: int, num_evaluations: int
    ) -> bool:
        self.timesteps_elapsed = timesteps_elapsed
        desired_num_evaluations = self.timesteps_elapsed // self.step_freq
        if not self.skip_evaluate_at_start:
            desired_num_evaluations += 1
        return desired_num_evaluations > num_evaluations

    @property
    @abstractmethod
    def best_eval_stats(self) -> Optional[EpisodesStats]:
        ...

    @abstractmethod
    def enqueue_eval(self, eval_data: EvalView) -> None:
        ...

    @abstractmethod
    def evaluate(
        self,
        eval_data: EvalView,
        n_episodes: Optional[int] = None,
        print_returns: Optional[bool] = None,
    ) -> EpisodesStats:
        ...

    @abstractmethod
    def evaluate_latest_policy(
        self,
        algorithm: "Algorithm",
        n_episodes: Optional[int] = None,
        print_returns: bool = False,
    ) -> EpisodesStats:
        ...

    @abstractmethod
    def save(self, policy: Policy, model_path: str) -> None:
        ...
