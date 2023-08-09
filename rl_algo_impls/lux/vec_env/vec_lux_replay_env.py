import os
from typing import Dict, Optional

import numpy as np
from gym.vector.vector_env import VectorEnv

from rl_algo_impls.lux.rewards import LuxRewardWeights
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
        **kwargs,
    ) -> None:
        self.num_envs = num_envs
        self.replay_dir = replay_dir
        self.team_name = team_name
        self.offset_env_starts = offset_env_starts

        self.replay_paths = []
        for dirpath, _, filenames in os.walk(replay_dir):
            for fname in filenames:
                basename, ext = os.path.splitext(fname)
                if ext != ".json" or not basename.isdigit():
                    continue
                self.replay_paths.append(os.path.join(dirpath, fname))
        self.next_replay_idx = 0

        self.envs = [
            LuxReplayEnv(self.next_replay_path, team_name, reward_weights, **kwargs)
            for _ in range(self.num_envs)
        ]
        single_env = self.envs[0]
        map_dim = single_env.state.map_size
        self.num_map_tiles = map_dim * map_dim
        single_observation_space = single_env.observation_space
        self.action_plane_space = single_env.action_plane_space
        single_action_space = single_env.action_space
        self.metadata = single_env.metadata
        super().__init__(num_envs, single_observation_space, single_action_space)

    def next_replay_path(self) -> str:
        rp = self.replay_paths[self.next_replay_idx]
        self.next_replay_idx = (self.next_replay_idx + 1) % len(self.replay_paths)
        return rp

    def step(self, action: np.ndarray) -> VecEnvStepReturn:
        step_returns = [env.step() for env in self.envs]
        obs = np.stack([sr[0] for sr in step_returns])
        rewards = np.stack([sr[1] for sr in step_returns])
        dones = np.stack([sr[2] for sr in step_returns])
        infos = [sr[3] for sr in step_returns]
        return obs, rewards, dones, infos

    def reset(self) -> VecEnvObs:
        env_observations = [env.reset() for env in self.envs]
        if self.offset_env_starts:
            max_episode_length = self.envs[0].state.env_cfg.max_episode_length
            for idx, env in enumerate(self.envs):
                offset = int(max_episode_length * idx / self.num_envs)
                for _ in range(offset):
                    env_observations[idx], _, _, _ = env.step()
        return np.stack(env_observations)

    def seed(self, seeds=None):
        pass

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
        return self.envs[0].reward_weights

    @reward_weights.setter
    def reward_weights(self, reward_weights: LuxRewardWeights) -> None:
        for env in self.envs:
            env.reward_weights = reward_weights
