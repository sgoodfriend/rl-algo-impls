import logging
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import ray
from gym import Env
from gym.spaces import Box
from gym.spaces import Dict as DictSpace
from gym.spaces import MultiDiscrete

from rl_algo_impls.lux.actions import ACTION_SIZES
from rl_algo_impls.lux.replay_stats import ReplayActionStats, UnitBuiltStats
from rl_algo_impls.lux.rewards import LuxRewardWeights
from rl_algo_impls.lux.vec_env.lux_replay_state import ReplayPath


class Replay(NamedTuple):
    obs: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    action: List[Dict[str, np.ndarray]]
    action_mask: List[Dict[str, np.ndarray]]


@ray.remote
def async_load_npz(file_path: str) -> Replay:
    return sync_load_npz(file_path)


def sync_load_npz(file_path: str) -> Replay:
    def load_action(
        data: Dict[str, np.ndarray], name: str
    ) -> List[Dict[str, np.ndarray]]:
        return [
            {
                "per_position": per,
                "pick_position": pick,
            }
            for per, pick in zip(
                data[f"{name}_per_position"], data[f"{name}_pick_position"]
            )
        ]

    with np.load(file_path, allow_pickle=True) as data:
        replay = Replay(
            data["obs"],
            data["reward"],
            data["done"],
            load_action(data, "actions"),
            load_action(data, "action_mask"),
        )
    return replay


class LuxNpzReplayEnv(Env):
    def __init__(
        self,
        next_npz_path_fn: Callable[[], ReplayPath],
        reward_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.next_npz_path_fn = next_npz_path_fn
        self.reward_weights = (
            LuxRewardWeights(score_vs_opponent=1)
            if reward_weights is None
            else LuxRewardWeights(**reward_weights)
        )
        self.reward_weights.assert_non_zero_fields(
            {
                "score_vs_opponent",
                "built_light_by_time_remaining",
                "built_heavy_by_time_remaining",
            }
        )

        self.initialized = False

        self._load_request = async_load_npz.remote(next_npz_path_fn().replay_path)

        self.action_stats = ReplayActionStats()

    def initialize(self) -> None:
        self._load_next_replay()
        self.map_size = self.obs.shape[-2]
        self.num_map_tiles = self.map_size * self.map_size
        self.action_plane_space = MultiDiscrete(ACTION_SIZES)
        self.action_space = DictSpace(
            {
                "per_position": MultiDiscrete(
                    np.array(ACTION_SIZES * self.num_map_tiles).flatten().tolist()
                ),
                "pick_position": MultiDiscrete([self.num_map_tiles]),
            }
        )
        self.action_mask_shape = {
            "per_position": (
                self.num_map_tiles,
                self.action_plane_space.nvec.sum(),
            ),
            "pick_position": (
                len(self.action_space["pick_position"].nvec),
                self.num_map_tiles,
            ),
        }

        self.observation_space = Box(
            low=0,
            high=1,
            shape=self.obs.shape[1:],
            dtype=np.float32,
        )
        self.initialized = True

    def reset(self) -> np.ndarray:
        assert self.initialized
        self.action_stats = ReplayActionStats()
        self._load_next_replay()
        self._action_mask = self.action_mask[self.env_step]
        return self.obs[self.env_step]

    def step(
        self, action: Optional[Dict[str, np.ndarray]]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert self.initialized
        score_reward = self.reward[self.env_step]
        d = self.done[self.env_step]
        last_unit_built_stats = (
            self._last_unit_built_stats if self.env_step > 0 else UnitBuiltStats()
        )
        self._last_action = self.action[self.env_step]
        self._last_unit_built_stats = self.action_stats.update_action_stats(
            self._last_action, action, self._action_mask
        )
        self.env_step += 1

        if d:
            if score_reward > 0:
                win_loss = 1
            elif score_reward < 0:
                win_loss = -1
            else:
                win_loss = 0
            info = {
                "stats": self.action_stats.stats_dict(),
                "results": {
                    "WinLoss": win_loss,
                    "win": score_reward > 0,
                    "loss": score_reward < 0,
                    "score_reward": score_reward,
                },
            }
            self.reset()
        else:
            assert score_reward == 0
            info = {}
        o = self.obs[self.env_step]
        self._action_mask = self.action_mask[self.env_step]

        game_remaining = 1 - o[0, 0, -3]  # Third-to-last channel is game progress
        weights = np.array(
            [
                self.reward_weights.score_vs_opponent,
                self.reward_weights.built_light_by_time_remaining,
                self.reward_weights.built_heavy_by_time_remaining,
            ]
        )
        raw_rewards = np.array(
            [
                score_reward,
                last_unit_built_stats.built_light * game_remaining,
                last_unit_built_stats.built_heavy * game_remaining,
            ]
        )
        r = weights.dot(raw_rewards)
        return (o, r, d, info)

    def get_action_mask(self) -> Dict[str, np.ndarray]:
        assert self.initialized
        return self._action_mask

    def close(self) -> None:
        self.next_npz_path_fn = None
        self._load_request = None

    @property
    def last_action(self) -> Dict[str, np.ndarray]:
        return self._last_action

    @property
    def max_episode_length(self) -> int:
        return 1000

    def _load_next_replay(self):
        assert (
            self.next_npz_path_fn
        ), f"next_npz_path_fn unset. Has {self.__class__.__name__} been closed?"
        if self._load_request:
            self.obs, self.reward, self.done, self.action, self.action_mask = ray.get(
                self._load_request
            )
        else:
            logging.warn("Synchronous load of npz file")
            (
                self.obs,
                self.reward,
                self.done,
                self.action,
                self.action_mask,
            ) = sync_load_npz(self.next_npz_path_fn().replay_path)
        self.env_step = 0
        self._load_request = async_load_npz.remote(self.next_npz_path_fn().replay_path)
