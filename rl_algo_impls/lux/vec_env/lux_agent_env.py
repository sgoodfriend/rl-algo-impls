from typing import Dict, Optional

import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import MultiDiscrete

from rl_algo_impls.lux.actions import ACTION_SIZES, SIMPLE_ACTION_SIZES
from rl_algo_impls.lux.agent_config import LuxAgentConfig
from rl_algo_impls.lux.kit.config import EnvConfig
from rl_algo_impls.lux.obs_feature import SIMPLE_OBSERVATION_FEATURE_LENGTH
from rl_algo_impls.shared.vec_env.base_vector_env import BaseVectorEnv


class LuxAgentEnv(BaseVectorEnv):
    def __init__(
        self,
        num_envs: int,
        env_cfg: EnvConfig,
        bid_std_dev: float = 5,
        **kwargs,
    ) -> None:
        self.env_cfg = env_cfg
        self.bid_std_dev = bid_std_dev
        self.agent_cfg = LuxAgentConfig.from_kwargs(**kwargs)

        self.map_size = env_cfg.map_size
        self.num_map_tiles = self.map_size * self.map_size

        action_sizes = (
            SIMPLE_ACTION_SIZES
            if self.agent_cfg.use_simplified_spaces
            else ACTION_SIZES
        )
        single_observation_space = Box(
            low=-1,
            high=1,
            shape=(SIMPLE_OBSERVATION_FEATURE_LENGTH, self.map_size, self.map_size),
            dtype=np.float32,
        )
        single_action_space = DictSpace(
            {
                "per_position": MultiDiscrete(
                    np.array(action_sizes * self.num_map_tiles).flatten().tolist()
                ),
                "pick_position": MultiDiscrete([self.num_map_tiles]),
            }
        )

        super().__init__(num_envs, single_observation_space, single_action_space)
        self.action_plane_space = MultiDiscrete(action_sizes)

        self.action_mask_shape = {
            "per_position": (
                self.num_map_tiles,
                self.action_plane_space.nvec.sum(),
            ),
            "pick_position": (
                len(self.single_action_space["pick_position"].nvec),
                self.num_map_tiles,
            ),
        }
