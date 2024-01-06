from dataclasses import dataclass
from typing import List

from rl_algo_impls.ppo.ppo import TrainStats, TrainStepStats


@dataclass
class APPOTrainStats(TrainStats):
    n_epochs: float

    def __init__(
        self,
        step_stats: List[TrainStepStats],
        explained_var: float,
        grad_norms: List[float],
        n_epochs: float,
    ) -> None:
        super().__init__(step_stats, explained_var, grad_norms)
        self.n_epochs = n_epochs
