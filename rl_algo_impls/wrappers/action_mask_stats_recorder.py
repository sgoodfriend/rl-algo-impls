from typing import Any, Dict, List, Optional

import numpy as np
from gym import Env

from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnvObs,
    VecEnvStepReturn,
    VectorableWrapper,
)


class ActionMaskStatsRecorder(VectorableWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.get_action_mask = getattr(env, "get_action_mask")

    def reset(self) -> VecEnvObs:
        obs = super().reset()
        self.episode_num_valid_locations = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_no_valid_actions = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs

    def step(self, actions) -> VecEnvStepReturn:
        obs, rews, dones, infos = super().step(actions)

        action_mask = self.get_action_mask()
        num_valid_locations = np.sum(np.any(action_mask, axis=2), axis=1)
        self.episode_num_valid_locations += num_valid_locations
        self.episode_no_valid_actions += num_valid_locations == 0
        self.episode_lengths += 1

        for idx, (info, done) in enumerate(zip(infos, dones)):
            if done:
                info["action_mask_stats"] = {
                    "valid_locs": self.episode_num_valid_locations[idx]
                    / self.episode_lengths[idx],
                    "no_valid": self.episode_no_valid_actions[idx]
                    / self.episode_lengths[idx],
                }
                self.episode_num_valid_locations[idx] = 0
                self.episode_no_valid_actions[idx] = 0
                self.episode_lengths[idx] = 0

        return obs, rews, dones, infos
