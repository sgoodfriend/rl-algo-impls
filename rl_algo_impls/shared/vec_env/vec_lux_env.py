from typing import List, Sequence, TypeVar

import gym
import numpy as np
from gym.vector.utils import batch_space
from stable_baselines3.common.vec_env.base_vec_env import tile_images

from rl_algo_impls.wrappers.lux_env_gridnet import DEFAULT_REWARD_WEIGHTS, LuxEnvGridnet
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnvObs, VecEnvStepReturn

VecLuxEnvSelf = TypeVar("VecLuxEnvSelf", bound="VecLuxEnv")


class VecLuxEnv:
    def __init__(
        self,
        num_envs: int,
        bid_std_dev: float = 5,
        reward_weight: Sequence[float] = DEFAULT_REWARD_WEIGHTS,
        **kwargs,
    ) -> None:
        assert num_envs % 2 == 0, f"{num_envs} must be even"
        self.num_envs = num_envs
        self.envs = [
            LuxEnvGridnet(
                gym.make("LuxAI_S2-v0", collect_stats=True, verbose=1, **kwargs),
                bid_std_dev=bid_std_dev,
                reward_weight=reward_weight,
            )
            for _ in range(num_envs // 2)
        ]
        self.is_vector_env = True
        single_env = self.envs[0]
        map_dim = single_env.unwrapped.env_cfg.map_size
        self.num_map_tiles = map_dim * map_dim
        self.single_observation_space = single_env.single_observation_space
        self.observation_space = batch_space(self.single_observation_space, n=num_envs)
        self.action_plane_space = single_env.action_plane_space
        self.single_action_space = single_env.single_action_space
        self.action_space = batch_space(self.single_action_space, n=num_envs)
        self.metadata = single_env.metadata

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

    def close(self):
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
