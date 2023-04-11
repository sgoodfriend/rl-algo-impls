import numpy as np

from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.callbacks import Callback
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv


class MicrortsRewardDecayCallback(Callback):
    def __init__(
        self,
        config: Config,
        env: VecEnv,
        start_timesteps: int = 0,
    ) -> None:
        super().__init__()
        from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

        unwrapped = env.unwrapped
        assert isinstance(unwrapped, MicroRTSGridModeVecEnv)
        self.microrts_env = unwrapped
        self.base_reward_weights = self.microrts_env.reward_weight

        self.total_train_timesteps = config.n_timesteps
        self.timesteps_elapsed = start_timesteps

    def on_step(self, timesteps_elapsed: int = 1) -> bool:
        super().on_step(timesteps_elapsed)

        progress = self.timesteps_elapsed / self.total_train_timesteps
        # Decay all rewards except WinLoss
        reward_weights = self.base_reward_weights * np.array(
            [1] + [1 - progress] * (len(self.base_reward_weights) - 1)
        )
        self.microrts_env.reward_weight = reward_weights

        return True
