import logging
from collections import deque
from typing import List, NamedTuple, Optional, Union

import numpy as np
import ray
from gym.vector.vector_env import VectorEnv
from stable_baselines3.common.vec_env.base_vec_env import tile_images

from rl_algo_impls.lux.rewards import LuxRewardWeights
from rl_algo_impls.lux.vec_env.lux_ray_env import LuxRayEnv
from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnvMaskedResetReturn,
    VecEnvObs,
    VecEnvStepReturn,
)

try:
    import ray

    ray.init(_system_config={"automatic_object_spilling_enabled": False})
except ImportError:
    raise ImportError("Please install ray to use this class: `pip install ray`")


class PendingReset(NamedTuple):
    env: ray.ObjectRef  # [LuxRayEnv]
    reset_future: ray.ObjectRef  # [LuxRayResetReturn]


class LuxRayVectorEnv(VectorEnv):
    def __init__(self, num_envs: int, **kwargs) -> None:
        assert num_envs % 2 == 0, f"{num_envs} must be even"
        assert num_envs > 2, "Use VecLuxEnv instead"

        self.all_envs = [LuxRayEnv.remote(**kwargs) for i in range(num_envs)]
        self.envs = self.all_envs[: num_envs // 2]
        self.pending_resets = deque(
            PendingReset(e, e.reset.remote()) for e in self.all_envs[num_envs // 2 :]
        )

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
        obs = []
        rewards = []
        dones = []
        infos = []
        action_masks = []
        reset_future_indexes = []
        reset_futures = []
        for idx, ((o, r, d, i), am) in enumerate(step_returns):
            if d.any():
                assert d.all(), "All dones should be True if any done is True"
                self.pending_resets.append(
                    PendingReset(self.envs[idx], self.envs[idx].reset.remote())
                )
                self.envs[idx], reset_future = self.pending_resets.popleft()
                reset_future_indexes.append(idx)
                reset_futures.append(reset_future)
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.extend(i)
            action_masks.append(am)
        reset_returns = ray.get(reset_futures)
        for idx, (o, am) in zip(reset_future_indexes, reset_returns):
            obs[idx] = o
            action_masks[idx] = am
        self._action_masks = np.concatenate(action_masks)
        return (
            np.concatenate(obs),
            np.concatenate(rewards),
            np.concatenate(dones),
            infos,
        )

    def reset(self) -> VecEnvObs:
        self.pending_resets.extend(
            [PendingReset(e, e.reset.remote()) for e in self.envs]
        )

        resets = [self.pending_resets.popleft() for _ in range(len(self.envs))]
        self.envs = [r.env for r in resets]
        reset_returns = ray.get([r.reset_future for r in resets])
        obs = np.concatenate([sr.obs for sr in reset_returns])
        self._action_masks = np.concatenate([sr.action_mask for sr in reset_returns])
        return obs

    def masked_reset(self, env_mask: np.ndarray) -> VecEnvMaskedResetReturn:
        assert np.all(
            env_mask[::2] == env_mask[1::2]
        ), f"Expect env_mask to be the same for player 1 and 2: {env_mask}"
        mapped_mask = env_mask[::2]
        reset_futures = []
        for idx, (env, m) in enumerate(zip(self.envs, mapped_mask)):
            if not m:
                continue
            self.pending_resets.append(PendingReset(env, env.reset.remote()))
            self.envs[idx], reset_future = self.pending_resets.popleft()
            reset_futures.append(reset_future)
        reset_returns = ray.get(reset_futures)
        obs = np.concatenate([sr.obs for sr in reset_returns])
        action_masks = np.concatenate([sr.action_mask for sr in reset_returns])
        self._action_masks[env_mask] = action_masks
        return VecEnvMaskedResetReturn(obs, action_masks)

    def seed(self, seed: Optional[int]) -> None:
        seed_rng = np.random.RandomState(seed)
        for e, s in zip(
            self.all_envs,
            seed_rng.randint(0, np.iinfo(np.int32).max, size=len(self.all_envs)),
        ):
            e.seed.remote(s)

    def close_extras(self, **kwargs):
        ray.get([e.close.remote() for e in self.all_envs])
        ray.shutdown()

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
