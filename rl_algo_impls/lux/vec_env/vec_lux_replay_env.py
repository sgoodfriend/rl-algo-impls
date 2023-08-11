import os
from typing import Dict, Optional, Tuple

import numpy as np
import ray
from gym.spaces import Box
from gym.spaces import Dict as DictSpace
from gym.spaces import MultiDiscrete
from gym.vector.vector_env import VectorEnv

from rl_algo_impls.lux.rewards import LuxRewardWeights
from rl_algo_impls.lux.vec_env.lux_npz_replay_env import LuxNpzReplayEnv
from rl_algo_impls.lux.vec_env.lux_replay_env import LuxReplayEnv
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnvObs, VecEnvStepReturn


class VecLuxReplayEnv(VectorEnv):
    def __init__(
        self,
        num_envs: int,
        replay_dir: str,
        team_name: str,
        reward_weights: Optional[Dict[str, float]] = None,
        offset_env_starts: bool = False,
        is_npz_dir: bool = False,
        **kwargs,
    ) -> None:
        self.num_envs = num_envs
        self._reward_weights = (
            LuxRewardWeights.default_start()
            if reward_weights is None
            else LuxRewardWeights(**reward_weights)
        )
        ray.init(_system_config={"automatic_object_spilling_enabled": False})
        self.actor = RemoteVecLuxReplayEnv.remote(
            num_envs,
            replay_dir,
            team_name,
            reward_weights,
            offset_env_starts,
            is_npz_dir,
            **kwargs,
        )
        (
            single_observation_space,
            single_action_space,
            self.action_plane_space,
            self.metadata,
        ) = ray.get(self.actor.get_env_properties.remote())
        super().__init__(num_envs, single_observation_space, single_action_space)

    def step(self, action: np.ndarray) -> VecEnvStepReturn:
        (obs, r, d, i), action_mask, self._last_action = ray.get(self._step_task)
        self._step_task = self.actor.step.remote()
        self._action_mask = action_mask.copy()
        return obs.copy(), r, d, i

    def reset(self) -> VecEnvObs:
        obs, action_mask = ray.get(self.actor.reset.remote())
        self._step_task = self.actor.step.remote()
        self._action_mask = action_mask.copy()
        return obs.copy()

    def seed(self, seeds=None):
        pass

    def close_extras(self, **kwargs):
        self.actor.close_extras.remote(**kwargs)
        ray.shutdown()

    def render(self, mode="human", **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support rendering"
        )

    def get_action_mask(self) -> np.ndarray:
        return self._action_mask

    @property
    def last_action(self) -> np.ndarray:
        return self._last_action

    @property
    def reward_weights(self) -> LuxRewardWeights:
        return self._reward_weights


@ray.remote(num_cpus=1)
class RemoteVecLuxReplayEnv:
    def __init__(
        self,
        num_envs: int,
        replay_dir: str,
        team_name: str,
        reward_weights: Optional[Dict[str, float]],
        offset_env_starts: bool,
        is_npz_dir: bool,
        **kwargs,
    ) -> None:
        self.num_envs = num_envs
        self.replay_dir = replay_dir
        self.team_name = team_name
        self.offset_env_starts = offset_env_starts
        self.is_npz_dir = is_npz_dir

        self.replay_paths = []
        for dirpath, _, filenames in os.walk(replay_dir):
            for fname in filenames:
                basename, ext = os.path.splitext(fname)
                if (
                    ext != (".npz" if self.is_npz_dir else ".json")
                    or not basename.isdigit()
                ):
                    continue
                self.replay_paths.append(os.path.join(dirpath, fname))
        self.next_replay_idx = 0

        if self.is_npz_dir:
            self.envs = [
                LuxNpzReplayEnv(self.next_replay_path) for _ in range(self.num_envs)
            ]
            for e in self.envs:
                e.initialize()
        else:
            self.envs = [
                LuxReplayEnv(self.next_replay_path, team_name, reward_weights, **kwargs)
                for _ in range(self.num_envs)
            ]

    def next_replay_path(self) -> str:
        rp = self.replay_paths[self.next_replay_idx]
        self.next_replay_idx = (self.next_replay_idx + 1) % len(self.replay_paths)
        return rp

    def step(self) -> Tuple[VecEnvStepReturn, np.ndarray, np.ndarray]:
        step_returns = [env.step(None) for env in self.envs]
        obs = np.stack([sr[0] for sr in step_returns])
        rewards = np.stack([sr[1] for sr in step_returns])
        dones = np.stack([sr[2] for sr in step_returns])
        infos = [sr[3] for sr in step_returns]
        return (obs, rewards, dones, infos), self.get_action_mask(), self.last_action

    def reset(self) -> Tuple[VecEnvObs, np.ndarray]:
        env_observations = [env.reset() for env in self.envs]
        if self.offset_env_starts:
            max_episode_length = self.envs[0].max_episode_length
            for idx, env in enumerate(self.envs):
                offset = int(max_episode_length * idx / self.num_envs)
                for _ in range(offset):
                    env_observations[idx], _, _, _ = env.step(None)
        return np.stack(env_observations), self.get_action_mask()

    def get_env_properties(self) -> Tuple[Box, DictSpace, MultiDiscrete, Dict]:
        single_env = self.envs[0]
        single_observation_space = single_env.observation_space
        action_plane_space = single_env.action_plane_space
        single_action_space = single_env.action_space
        metadata = single_env.metadata
        return (
            single_observation_space,
            single_action_space,
            action_plane_space,
            metadata,
        )

    def close_extras(self, **kwargs):
        for env in self.envs:
            env.close()

    def render(self, mode="human", **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support rendering"
        )

    def get_action_mask(self) -> np.ndarray:
        return np.stack([env.get_action_mask() for env in self.envs])

    @property
    def last_action(self) -> np.ndarray:
        return np.stack([env.last_action for env in self.envs])

    @property
    def reward_weights(self) -> LuxRewardWeights:
        assert hasattr(self.envs[0], "reward_weights")
        return self.envs[0].reward_weights  # type: ignore

    @reward_weights.setter
    def reward_weights(self, reward_weights: LuxRewardWeights) -> None:
        assert hasattr(self.envs[0], "reward_weights")
        for env in self.envs:
            env.reward_weights = reward_weights  # type: ignore
