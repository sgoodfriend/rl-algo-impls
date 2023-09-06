from typing import List, Optional, Union

import numpy as np
import ray
from gym.vector.vector_env import VectorEnv
from stable_baselines3.common.vec_env.base_vec_env import tile_images

from rl_algo_impls.lux.rewards import LuxRewardWeights
from rl_algo_impls.lux.vec_env.lux_ray_env import LuxRayEnv
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnvObs, VecEnvStepReturn

try:
    import ray

    ray.init(_system_config={"automatic_object_spilling_enabled": False})
except ImportError:
    raise ImportError("Please install ray to use this class: `pip install ray`")


class LuxRayVectorEnv(VectorEnv):
    def __init__(self, num_envs: int, **kwargs) -> None:
        assert num_envs % 2 == 0, f"{num_envs} must be even"
        assert num_envs > 2, "Use VecLuxEnv instead"

        self.envs = [LuxRayEnv.remote(**kwargs) for _ in range(num_envs // 2)]

        (
            map_dim,
            single_observation_space,
            single_action_space,
            self.action_plane_space,
            self.metadata,
            self._reward_weights,
        ) = ray.get(self.envs[0].get_properties.remote())
        self.num_map_tiles = map_dim * map_dim
        super().__init__(num_envs, single_observation_space, single_action_space)

    def step(self, action: np.ndarray) -> VecEnvStepReturn:
        step_returns = ray.get(
            [
                env.step.remote(action[2 * idx : 2 * idx + 2])
                for idx, env in enumerate(self.envs)
            ]
        )
        obs = np.concatenate([sr.step_return[0] for sr in step_returns])
        rewards = np.concatenate([sr.step_return[1] for sr in step_returns])
        dones = np.concatenate([sr.step_return[2] for sr in step_returns])
        infos = [info for sr in step_returns for info in sr.step_return[3]]
        self._action_masks = np.concatenate([sr.action_mask for sr in step_returns])
        return obs, rewards, dones, infos

    def reset(self) -> VecEnvObs:
        reset_returns = ray.get([env.reset.remote() for env in self.envs])
        obs = np.concatenate([sr.obs for sr in reset_returns])
        self._action_masks = np.concatenate([sr.action_mask for sr in reset_returns])
        return obs

    def seed(self, seeds: Optional[Union[int, List[int]]]):
        if seeds is None:
            _seeds = [None] * len(self.envs)
        elif isinstance(seeds, int):
            _seeds = [seeds + i for i in range(len(self.envs))]
        else:
            _seeds = seeds
        for e, s in zip(self.envs, _seeds):
            e.seed.remote(s)

    def close_extras(self, **kwargs):
        ray.shutdown()
        return super().close_extras(**kwargs)

    def render(self, mode: str = "human", **kwargs):
        if mode == "human":
            for e in self.envs:
                e.render.remote(mode, **kwargs)
        elif mode == "rgb_array":
            imgs = ray.get([e.render.remote(mode, **kwargs) for e in self.envs])
            bigimg = tile_images(imgs)
            return bigimg
        else:
            raise ValueError(
                f"Unknown mode {mode}. Only 'human' and 'rgb_array' are supported"
            )

    def get_action_mask(self) -> np.ndarray:
        return self._action_masks

    @property
    def reward_weights(self) -> LuxRewardWeights:
        return self._reward_weights

    @reward_weights.setter
    def reward_weights(self, reward_weights: LuxRewardWeights) -> None:
        self._reward_weights = reward_weights
        for e in self.envs:
            e.set_reward_weights.remote(reward_weights)
