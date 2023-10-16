from typing import Any, Dict, List, Optional

import numpy as np
from gymnasium import Env

from rl_algo_impls.wrappers.vector_wrapper import (
    VecEnvResetReturn,
    VecEnvStepReturn,
    VectorWrapper,
)


class ActionMaskStatsRecorder(VectorWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.get_action_mask = getattr(env, "get_action_mask")

    def reset(self, **kwargs) -> VecEnvResetReturn:
        reset_return = super().reset(**kwargs)
        self.episode_num_valid_locations = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_no_valid_actions = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return reset_return

    def step(self, actions) -> VecEnvStepReturn:
        obs, rewards, terminations, truncations, infos = super().step(actions)
        dones = terminations | truncations

        action_mask = self.get_action_mask()
        num_valid_locations = np.sum(np.any(action_mask, axis=2), axis=1)
        self.episode_num_valid_locations += num_valid_locations
        self.episode_no_valid_actions += num_valid_locations == 0
        self.episode_lengths += 1

        for idx, done in enumerate(dones):
            if done:
                self._add_info(
                    infos,
                    {
                        "action_mask_stats": {
                            "valid_locs": self.episode_num_valid_locations[idx]
                            / self.episode_lengths[idx],
                            "no_valid": self.episode_no_valid_actions[idx]
                            / self.episode_lengths[idx],
                        }
                    },
                    idx,
                )
                self.episode_num_valid_locations[idx] = 0
                self.episode_no_valid_actions[idx] = 0
                self.episode_lengths[idx] = 0

        return obs, rewards, terminations, truncations, infos
