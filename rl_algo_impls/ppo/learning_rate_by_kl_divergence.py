from typing import Optional

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
        v_loss_threshold: Optional[float] = None,
        v_loss_fast_moving_window_size: int = 10,
        v_loss_slow_moving_window_size: int = 50,
        no_increase_on_max_grad_norm: bool = False,
        min_lr: Optional[float] = None,
        max_lr: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.algo = algo
        self.target_kl = target_kl
        self.max_increase_fraction = max_increase_fraction
        self.min_decrease_fraction = min_decrease_fraction

        self.rms_kl = ExponentialMovingMeanVar(
            "lr_by_kl.npz", window_size=moving_window_size
        )
        self.num_updates = 0

        self.v_loss_threshold = v_loss_threshold
        if v_loss_threshold is not None:
            self.slow_v_loss_rms = ExponentialMovingMeanVar(
                window_size=v_loss_slow_moving_window_size,
            )
            self.fast_v_loss_rms = ExponentialMovingMeanVar(
                window_size=v_loss_fast_moving_window_size,
            )

        self.no_increase_on_max_grad_norm = no_increase_on_max_grad_norm

        self.min_lr = min_lr
        self.max_lr = max_lr
        if self.min_lr is not None:
            assert (
                self.algo.learning_rate >= self.min_lr
            ), f"Algo's learning rate is already below min_lr"
        if self.max_lr is not None:
            assert (
                self.algo.learning_rate <= self.max_lr
            ), f"Algo's learning rate is already above max_lr"

    def on_step(
        self, train_stats: TrainStats, timesteps_elapsed: int = 1, **kwargs
    ) -> bool:
        super().on_step(timesteps_elapsed)

        self.rms_kl.update(np.array([train_stats.approx_kl]))
        self.num_updates += 1

        if self.v_loss_threshold is not None:
            v_loss = np.array([np.mean(train_stats.v_loss)])
            self.slow_v_loss_rms.update(v_loss)
            self.fast_v_loss_rms.update(v_loss)

        min_decrease_fraction = self.min_decrease_fraction
        max_increase_fraction = self.max_increase_fraction
        if (
            self.v_loss_threshold is not None
            and self.num_updates > self.slow_v_loss_rms.window_size
        ):
            v_loss_ratio = (
                self.fast_v_loss_rms.mean.item() / self.slow_v_loss_rms.mean.item()
            )
            if v_loss_ratio > self.v_loss_threshold:
                max_increase_fraction -= v_loss_ratio - self.v_loss_threshold
                max_increase_fraction = max(
                    max_increase_fraction, min_decrease_fraction
                )
        if (
            self.no_increase_on_max_grad_norm
            and train_stats.grad_norm > self.algo.max_grad_norm
        ):
            max_increase_fraction = min(max_increase_fraction, 1.0)

        if self.num_updates > self.rms_kl.window_size:
            kl = self.rms_kl.mean.item()
            correction_ratio = np.clip(
                self.target_kl / np.abs(kl),
                min_decrease_fraction,
                max_increase_fraction,
            )
            self.algo.learning_rate *= correction_ratio

        if self.min_lr is not None:
            self.algo.learning_rate = max(self.algo.learning_rate, self.min_lr)
        if self.max_lr is not None:
            self.algo.learning_rate = min(self.algo.learning_rate, self.max_lr)

        return True
