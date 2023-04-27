from typing import Dict, List, Optional, TypeVar

import gym
import numpy as np
from gym.vector.vector_env import VectorEnv
from stable_baselines3.common.vec_env.base_vec_env import tile_images

from rl_algo_impls.wrappers.lux_env_gridnet import LuxEnvGridnet, LuxRewardWeights
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnvObs, VecEnvStepReturn

VecLuxEnvSelf = TypeVar("VecLuxEnvSelf", bound="VecLuxEnv")


class VecLuxEnv(VectorEnv):
    def __init__(
        self,
        num_envs: int,
        bid_std_dev: float = 5,
        reward_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        assert num_envs % 2 == 0, f"{num_envs} must be even"
        self.envs = [
            LuxEnvGridnet(
                gym.make("LuxAI_S2-v0", collect_stats=True, **kwargs),
                bid_std_dev=bid_std_dev,
                reward_weights=reward_weights,
            )
            for _ in range(num_envs // 2)
        ]
        single_env = self.envs[0]
        map_dim = single_env.unwrapped.env_cfg.map_size
        self.num_map_tiles = map_dim * map_dim
        single_observation_space = single_env.single_observation_space
        self.action_plane_space = single_env.action_plane_space
        single_action_space = single_env.single_action_space
        self.metadata = single_env.metadata
        super().__init__(num_envs, single_observation_space, single_action_space)

    def step(self, action: np.ndarray) -> VecEnvStepReturn:
        step_returns = [
            env.step(action[2 * idx : 2 * idx + 2]) for idx, env in enumerate(self.envs)
        ]
        obs = np.concatenate([sr[0] for sr in step_returns])
        rewards = np.concatenate([sr[1] for sr in step_returns])
        dones = np.concatenate([sr[2] for sr in step_returns])
        infos = [info for sr in step_returns for info in sr[3]]
        return obs, rewards, dones, infos

    def reset(self) -> VecEnvObs:
        env_obervations = [env.reset() for env in self.envs]
        return np.concatenate(env_obervations)

    def seed(self, seeds=None):
        # TODO: Seeds aren't supported in LuxAI_S2
        pass

    def close_extras(self, **kwargs):
        for env in self.envs:
            env.close()

    @property
    def unwrapped(self: VecLuxEnvSelf) -> VecLuxEnvSelf:
        return self

    def render(self, mode="human", **kwargs):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode, **kwargs)
        if mode == "human":
            for env in self.envs:
                env.render(mode=mode, **kwargs)
        elif mode == "rgb_array":
            imgs = self.get_images()
            bigimg = tile_images(imgs)
            return bigimg

    def get_images(self) -> List[np.ndarray]:
        return [env.render(mode="rgb_array") for env in self.envs]

    def get_action_mask(self) -> np.ndarray:
        return np.concatenate([env.get_action_mask() for env in self.envs])

    @property
    def reward_weights(self) -> LuxRewardWeights:
        return self.envs[0].reward_weights

    @reward_weights.setter
    def reward_weights(self, reward_weights: LuxRewardWeights) -> None:
        for env in self.envs:
            env.reward_weights = reward_weights
