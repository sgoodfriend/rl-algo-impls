import gym
import numpy as np

from rl_algo_impls.wrappers.vectorable_wrapper import VecEnvMaskedResetReturn


class MaskResettableEpisodeStatistics(gym.wrappers.RecordEpisodeStatistics):
    def masked_reset(self, env_mask: np.ndarray) -> VecEnvMaskedResetReturn:
        self.episode_returns[env_mask] = 0
        self.episode_lengths[env_mask] = 0
        return self.env.masked_reset(env_mask)
