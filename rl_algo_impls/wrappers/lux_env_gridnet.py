from typing import Optional

import numpy as np
from gym.spaces import Box
from gym.spaces import Dict as GymDict
from gym.spaces import MultiDiscrete, Space
from luxai_s2.env import LuxAI_S2

from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnvObs,
    VecEnvStepReturn,
    VectorableWrapper,
)


class LuxEnvGridnet(VectorableWrapper):
    def __init__(self, env, bid_std_dev: int = 5) -> None:
        super().__init__(env)  # type: ignore
        self.bid_std_dev = bid_std_dev

    @property
    def unwrapped(self) -> LuxAI_S2:
        unwrapped = super().unwrapped
        assert isinstance(unwrapped, LuxAI_S2)
        return unwrapped

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        lux_actions = self._to_lux_actions(actions)
        lux_obs, lux_rewards, dones, infos = self.env.step(lux_actions)

        obs = self._from_lux_observation(lux_obs)
        rewards = self._from_lux_rewards(lux_rewards)

        return obs, rewards, dones, infos

    def reset(self) -> VecEnvObs:
        lux_obs = self.env.reset()

    def _from_lux_observation(self, lux_obs: GymDict) -> Box:
        env = self.unwrapped
        map_size = env.env_cfg.map_size

        p1_obs = lux_obs[env.agents[0]]

        x = np.tile(np.linspace(-1, 1, num=map_size), (map_size, 1))
        y = np.transpose(np.tile(np.linspace(-1, 1, num=map_size), (map_size, 1)))

        ore = p1_obs["board"]["ore"]
        ice = p1_obs["board"]["ice"]

        _rubble = p1_obs["board"]["rubble"]
        non_zero_rubble = _rubble > 0
        rubble = _rubble / env.env_cfg.MAX_RUBBLE

        _lichen = p1_obs["board"]["lichen"]
        non_zero_lichen = _lichen > 0
        lichen = _lichen / env.env_cfg.MAX_LICHEN_PER_TILE
        spreadable_lichen = _lichen >= env.env_cfg.MIN_LICHEN_TO_SPREAD
