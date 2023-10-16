import gymnasium
import numpy as np
from gymnasium.experimental.wrappers.vector.record_episode_statistics import (
    RecordEpisodeStatisticsV0,
)

from rl_algo_impls.wrappers.vector_wrapper import VecEnvMaskedResetReturn


class MaskResettableEpisodeStatistics(RecordEpisodeStatisticsV0):
    def masked_reset(self, env_mask: np.ndarray) -> VecEnvMaskedResetReturn:
        self.episode_returns[env_mask] = 0
        self.episode_lengths[env_mask] = 0
        return self.env.masked_reset(env_mask)  # type: ignore
