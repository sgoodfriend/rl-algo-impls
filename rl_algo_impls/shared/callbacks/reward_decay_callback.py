from typing import Iterable, Optional

import numpy as np

from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.callbacks import Callback
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv


class RewardDecayCallback(Callback):
    def __init__(
        self,
        config: Config,
        env: VecEnv,
        start_timesteps: int = 0,
        constant_indexes: Optional[Iterable[int]] = None,
        increase_indexes: Optional[Iterable[int]] = None,
    ) -> None:
        super().__init__()

        self.unwrapped = env.unwrapped
        assert hasattr(
            self.unwrapped, "reward_weight"
        ), "Env must have settable property reward_weight"
        base_reward_weights = getattr(self.unwrapped, "reward_weight")
        self.base_reward_weights = base_reward_weights

        self.total_train_timesteps = config.n_timesteps
        self.timesteps_elapsed = start_timesteps

        if not constant_indexes and not increase_indexes:
            constant_indexes = (0,)
            increase_indexes = tuple()
        self.constant_indexes = set(constant_indexes or tuple())
        self.increase_indexes = set(increase_indexes or tuple())
        self.on_step(timesteps_elapsed=0)

    def on_step(self, timesteps_elapsed: int = 1) -> bool:
        super().on_step(timesteps_elapsed)

        progress = self.timesteps_elapsed / self.total_train_timesteps
        reward_weights = []
        for i in range(len(self.base_reward_weights)):
            base_weight = self.base_reward_weights[i]
            if i in self.constant_indexes:
                reward_weights.append(base_weight)
            elif i in self.increase_indexes:
                reward_weights.append(base_weight * progress)
            else:
                reward_weights.append(base_weight * (1 - progress))
        setattr(self.unwrapped, "reward_weight", np.array(reward_weights))

        return True
