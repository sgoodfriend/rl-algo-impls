from abc import ABC, abstractmethod
from typing import Optional

from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.stats import EpisodesStats


class AbstractEvaluator(ABC):
    @abstractmethod
    def on_timesteps_elapsed(self, timesteps_elapsed: int) -> bool:
        ...

    @abstractmethod
    def close(self) -> Optional[EpisodesStats]:
        ...

    @abstractmethod
    def evaluate(
        self, n_episodes: Optional[int] = None, print_returns: Optional[bool] = None
    ) -> EpisodesStats:
        ...

    @abstractmethod
    def generate_video(self, policy: Policy) -> None:
        ...

    @abstractmethod
    def save(self, policy: Policy, model_path: str) -> None:
        ...
