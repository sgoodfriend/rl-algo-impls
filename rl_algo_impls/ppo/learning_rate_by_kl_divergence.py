import numpy as np

from rl_algo_impls.ppo.ppo import PPO, TrainStats
from rl_algo_impls.shared.callbacks.callback import Callback
from rl_algo_impls.utils.running_mean_std import ExponentialMovingMeanVar


class LearningRateByKLDivergence(Callback):
    def __init__(
        self,
        algo: PPO,
        target_kl: float,
        moving_window_size: int = 5,
        max_increase_fraction: float = 1.02,
        min_decrease_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        self.algo = algo
        self.target_kl = target_kl
        self.moving_window_size = moving_window_size
        self.max_increase_fraction = max_increase_fraction
        self.min_decrease_fraction = min_decrease_fraction

        self.rms_kl = ExponentialMovingMeanVar(window_size=moving_window_size)
        self.rms_updates = 0

    def on_step(
        self, train_stats: TrainStats, timesteps_elapsed: int = 1, **kwargs
    ) -> bool:
        super().on_step(timesteps_elapsed)

        self.rms_kl.update(np.array(train_stats.approx_kl))
        self.rms_updates += 1

        if self.rms_updates > self.moving_window_size:
            kl = self.rms_kl.mean.item()
            correction_ratio = np.clip(
                self.target_kl / kl,
                self.min_decrease_fraction,
                self.max_increase_fraction,
            )
            self.algo.learning_rate *= correction_ratio

        return True
