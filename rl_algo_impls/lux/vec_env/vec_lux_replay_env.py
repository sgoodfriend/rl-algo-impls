import logging
import os
from typing import Dict, List, NamedTuple, Optional

import numpy as np

from rl_algo_impls.lux.rewards import LuxRewardWeights
from rl_algo_impls.lux.vec_env.lux_npz_replay_env import LuxNpzReplayEnv
from rl_algo_impls.lux.vec_env.lux_replay_env import LuxReplayEnv
from rl_algo_impls.lux.vec_env.lux_replay_state import ReplayPath
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.shared.vec_env.base_vector_env import BaseVectorEnv
from rl_algo_impls.wrappers.vector_wrapper import (
    VecEnvResetReturn,
    VecEnvStepReturn,
    merge_info,
)


class ReplayDir(NamedTuple):
    replay_dir: str
    team_name: str


class VecLuxReplayEnv(BaseVectorEnv):
    def __init__(
        self,
        num_envs: int,
        tb_writer: Optional[AbstractSummaryWrapper],
        replay_dirs: List[Dict[str, str]],
        reward_weights: Optional[Dict[str, float]] = None,
        offset_env_starts: bool = False,
        is_npz_dir: bool = False,
        compare_policy_action: bool = False,
        **kwargs,
    ) -> None:
        self.num_envs = num_envs

        self.replay_dirs = [ReplayDir(**rd) for rd in replay_dirs]
        self.offset_env_starts = offset_env_starts
        self.is_npz_dir = is_npz_dir
        self.compare_policy_action = compare_policy_action

        self.replay_paths = []
        for replay_dir, team_name in self.replay_dirs:
            added_replay = False
            for dirpath, _, filenames in os.walk(replay_dir):
                for fname in filenames:
                    basename, ext = os.path.splitext(fname)
                    if (
                        ext != (".npz" if self.is_npz_dir else ".json")
                        or not basename.isdigit()
                    ):
                        continue
                    self.replay_paths.append(
                        ReplayPath(
                            replay_path=os.path.join(dirpath, fname),
                            team_name=team_name,
                        )
                    )
                    added_replay = True
            if not added_replay:
                logging.warn(f"Could not find any replays in {replay_dir}")
        self.next_replay_idx = 0
        self.replay_idx_permutation = np.random.permutation(len(self.replay_paths))

        if tb_writer:
            tb_writer.update_summary({"num_replays": len(self.replay_paths)})

        if self.is_npz_dir:
            import ray

            ray.init(
                _system_config={"automatic_object_spilling_enabled": False},
                ignore_reinit_error=True,
            )
            self.envs = [
                LuxNpzReplayEnv(self.next_replay_path, reward_weights)
                for _ in range(self.num_envs)
            ]
            for e in self.envs:
                e.initialize()
        else:
            self.envs = [
                LuxReplayEnv(self.next_replay_path, reward_weights, **kwargs)
                for _ in range(self.num_envs)
            ]
        single_env = self.envs[0]
        map_dim = single_env.map_size
        self.num_map_tiles = map_dim * map_dim
        self.action_plane_space = single_env.action_plane_space
        self.metadata = single_env.metadata
        super().__init__(
            num_envs, single_env.observation_space, single_env.action_space
        )

    def next_replay_path(self) -> ReplayPath:
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
        terminations = np.stack([sr[2] for sr in step_returns])
        truncations = np.stack([sr[3] for sr in step_returns])
        infos = merge_info(self, [sr[4] for sr in step_returns])
        return obs, rewards, terminations, truncations, infos

    def reset(self, **kwargs) -> VecEnvResetReturn:
        reset_returns = [env.reset(**kwargs) for env in self.envs]
        env_observations = [rr[0] for rr in reset_returns]
        infos = [rr[1] for rr in reset_returns]
        if self.offset_env_starts:
            max_episode_length = self.envs[0].max_episode_length
            for idx, env in enumerate(self.envs):
                offset = int(max_episode_length * idx / self.num_envs)
                for _ in range(offset):
                    env_observations[idx], _, _, _, infos[idx] = env.step(None)
        return np.stack(env_observations), merge_info(self, infos)

    def close_extras(self, **kwargs):
        if self.is_npz_dir:
            import ray

            ray.shutdown()
        for env in self.envs:
            env.close(**kwargs)

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

    def call(self, method_name: str, *args, **kwargs) -> tuple:
        results = []
        for env in self.envs:
            fn = getattr(env, method_name)
            if callable(fn):
                results.append(fn(*args, **kwargs))
            else:
                results.append(fn)
        return tuple(results)
