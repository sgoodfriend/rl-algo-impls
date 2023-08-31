import logging
import os
from typing import Dict, Optional

import numpy as np
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
        compare_policy_action: bool = False,
        **kwargs,
    ) -> None:
        self.num_envs = num_envs
        self.replay_dir = replay_dir
        self.team_name = team_name
        self.offset_env_starts = offset_env_starts
        self.is_npz_dir = is_npz_dir
        self.compare_policy_action = compare_policy_action

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
        self.replay_idx_permutation = np.random.permutation(len(self.replay_paths))

        try:
            import wandb

            if wandb.run:
                wandb.run.summary["num_replays"] = len(self.replay_paths)
        except ImportError:
            logging.warn("No wandb package. Not recording num_replays")

        if self.is_npz_dir:
            import ray

            ray.init(_system_config={"automatic_object_spilling_enabled": False})
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
        single_env = self.envs[0]
        map_dim = single_env.map_size
        self.num_map_tiles = map_dim * map_dim
        single_observation_space = single_env.observation_space
        self.action_plane_space = single_env.action_plane_space
        single_action_space = single_env.action_space
        self.metadata = single_env.metadata
        super().__init__(num_envs, single_observation_space, single_action_space)

    def next_replay_path(self) -> str:
        rp = self.replay_paths[self.replay_idx_permutation[self.next_replay_idx]]
        self.next_replay_idx += 1
        if self.next_replay_idx == len(self.replay_idx_permutation):
            self.replay_idx_permutation = np.random.permutation(len(self.replay_paths))
            self.next_replay_idx = 0
        return rp

    def step(self, action: np.ndarray) -> VecEnvStepReturn:
        if self.compare_policy_action:
            step_returns = [env.step(a) for env, a in zip(self.envs, action)]
        else:
            step_returns = [env.step(None) for env in self.envs]
        obs = np.stack([sr[0] for sr in step_returns])
        rewards = np.stack([sr[1] for sr in step_returns])
        dones = np.stack([sr[2] for sr in step_returns])
        infos = [sr[3] for sr in step_returns]
        return obs, rewards, dones, infos

    def reset(self) -> VecEnvObs:
        env_observations = [env.reset() for env in self.envs]
        if self.offset_env_starts:
            max_episode_length = self.envs[0].max_episode_length
            for idx, env in enumerate(self.envs):
                offset = int(max_episode_length * idx / self.num_envs)
                for _ in range(offset):
                    env_observations[idx], _, _, _ = env.step(None)
        return np.stack(env_observations)

    def seed(self, seeds=None):
        pass

    def close_extras(self, **kwargs):
        if self.is_npz_dir:
            import ray

            ray.shutdown()
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
