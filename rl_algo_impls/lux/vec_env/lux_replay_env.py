import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from gym import Env
from gym.spaces import Box
from gym.spaces import Dict as DictSpace
from gym.spaces import MultiDiscrete

from rl_algo_impls.lux.actions import (
    ACTION_SIZES,
    FACTORY_ACTION_ENCODED_SIZE,
    UNIT_ACTION_SIZES,
)
from rl_algo_impls.lux.kit.kit import GameState
from rl_algo_impls.lux.observation import observation_and_action_mask
from rl_algo_impls.lux.rewards import MIN_SCORE, LuxRewardWeights
from rl_algo_impls.lux.shared import pos_to_idx
from rl_algo_impls.lux.vec_env.lux_replay_state import LuxReplayState, LuxState


class LuxReplayEnv(Env):
    def __init__(
        self,
        next_replay_path_fn: Callable[[], str],
        team_name: str,
        reward_weights: Optional[Dict[str, float]] = None,
        verify: bool = False,
    ) -> None:
        super().__init__()
        self.next_replay_path_fn = next_replay_path_fn
        self.team_name = team_name
        self.reward_weights = (
            LuxRewardWeights.default_start()
            if reward_weights is None
            else LuxRewardWeights(**reward_weights)
        )
        self.verify = verify

        self.state = LuxReplayState(self.next_replay_path_fn(), self.team_name)
        obs_state_dict, game_state, enqueued_actions, player = self.state.reset(
            self.next_replay_path_fn()
        )
        # state.map_size not set until state.reset(...) is called
        self.map_size = self.state.map_size

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

        obs, _ = observation_and_action_mask(
            player,
            obs_state_dict,
            game_state,
            self.action_mask_shape,
            enqueued_actions,
            factory_ice_distance_buffer=None,
        )

        single_obs_shape = obs.shape
        # TODO: Verify low and high ranges are correct
        self.observation_space = Box(
            low=0,
            high=1,
            shape=single_obs_shape,
            dtype=np.float32,
        )

    def step(
        self,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        state, action, done, self_score, opponent_score = self.state.step()
        # _last_action must be set before _from_lux_state is called, which updates prior obs
        self._last_action = from_lux_action(
            self.action_space,
            state.player,
            action,
            self._action_mask,
            self._last_game_state,
        )
        obs = self._from_lux_state(state)

        score_delta = self_score - opponent_score
        reward = score_delta / (self_score + opponent_score + 1 - 2 * MIN_SCORE)
        if done:
            if self_score > opponent_score:
                win_loss = 1
            elif self_score < opponent_score:
                win_loss = -1
            else:
                win_loss = 0
            info = {
                "results": {
                    "WinLoss": win_loss,
                    "win": win_loss == 1,
                    "loss": win_loss == -1,
                    "score": self_score,
                    "score_delta": score_delta,
                    "score_reward": reward,
                }
            }
            obs = self.reset()
        else:
            info = {}
        return (obs, reward, done, info)

    def reset(self) -> np.ndarray:
        lux_state = self.state.reset(self.next_replay_path_fn())
        obs = self._from_lux_state(lux_state)
        return obs

    def _from_lux_state(self, lux_state: LuxState) -> np.ndarray:
        (obs_state_dict, game_state, enqueued_actions, player) = lux_state
        self._last_game_state = game_state
        obs, self._action_mask = observation_and_action_mask(
            player,
            obs_state_dict,
            game_state,
            self.action_mask_shape,
            enqueued_actions,
            factory_ice_distance_buffer=None,
        )
        return obs

    def get_action_mask(self) -> Dict[str, np.ndarray]:
        return self._action_mask

    def close(self) -> None:
        self.next_replay_path_fn = None

    @property
    def last_action(self) -> Dict[str, np.ndarray]:
        return self._last_action


def from_lux_action(
    action_space: DictSpace,
    player: str,
    lux_action: Dict[str, Union[int, str, List[List[int]]]],
    action_mask: Dict[str, np.ndarray],
    game_state: GameState,
) -> Dict[str, np.ndarray]:
    num_map_tiles = len(action_space["per_position"].nvec) // len(ACTION_SIZES)
    map_size = int(np.sqrt(num_map_tiles))
    assert (
        map_size * map_size == num_map_tiles
    ), f"Assuming square map, not {map_size}, {num_map_tiles / map_size}"
    action = {
        "per_position": np.zeros(
            (
                num_map_tiles,
                len(ACTION_SIZES),
            ),
            dtype=np.int32,
        ),
        "pick_position": np.zeros(
            len(action_space["pick_position"].nvec), dtype=np.int32
        ),
    }
    # Bid action
    if "bid" in lux_action:
        # TODO: Handle bid action
        return action
    # Factory placement action
    if "spawn" in lux_action:
        assert isinstance(lux_action["spawn"], list)
        pos = np.array(lux_action["spawn"])
        action["pick_position"][0] = pos_to_idx(pos, map_size)
        assert action_mask["pick_position"][0, pos_to_idx(pos, map_size)]
        # TODO: Handle metal and water assignment
        return action
    # Bid or factory placement phase. No unit or factory actions.
    if game_state.real_env_steps < 0:
        assert not my_turn_to_place_factory(
            game_state.teams[player].place_first, game_state.env_steps
        )
        return action
    for fu_id, a in lux_action.items():
        if fu_id.startswith("factory"):
            assert isinstance(a, int)
            factory = game_state.factories[player][fu_id]
            pos_idx = pos_to_idx(factory.pos, map_size)
            factory_mask = action_mask["per_position"][
                pos_idx, :FACTORY_ACTION_ENCODED_SIZE
            ]
            if not factory_mask[a]:
                logging.warn(
                    f"Invalid factory action {a} for factory {factory}. Mask {factory_mask}"
                )
                continue
            action["per_position"][pos_idx, 0] = a + 1
        else:
            unit = game_state.units[player][fu_id]
            pos_idx = pos_to_idx(unit.pos, map_size)
            assert isinstance(a, list)
            a_type, a_dir, a_resource_type, a_amount, a_repeat, a_n = a[0]
            pos_mask = action_mask["per_position"][pos_idx]
            unit_masks = []
            idx = FACTORY_ACTION_ENCODED_SIZE
            for mask_sz in UNIT_ACTION_SIZES:
                unit_masks.append(pos_mask[idx : idx + mask_sz])
                idx += mask_sz
            if not unit_masks[0][a_type]:
                logging.warn(
                    f"Invalid action type {a_type} for unit {unit}. Mask: {unit_masks[0]}"
                )
                continue
            _a = np.zeros(len(UNIT_ACTION_SIZES), dtype=np.int32)
            _a[0] = a_type
            if a_type == 0:
                if not unit_masks[1][a_dir]:
                    logging.warn(
                        f"Invalid move dir {a_dir} for unit {unit}. Mask: {unit_masks[1]}"
                    )
                    continue
                _a[1] = a_dir
            elif a_type == 1:
                if not unit_masks[2][a_dir]:
                    logging.warn(
                        f"Invalid transfer dir {a_dir} for unit {unit}. Mask: {unit_masks[2]}"
                    )
                    continue
                if not unit_masks[3][a_resource_type]:
                    logging.warn(
                        f"Invalid transfer resource {a_resource_type} for unit {unit}. Mask: {unit_masks[3]}"
                    )
                    continue
                _a[2] = a_dir
                _a[3] = a_resource_type
            elif a_type == 2:
                if not unit_masks[4][a_resource_type]:
                    logging.warn(
                        f"Invalid pickup resource {a_resource_type} for unit {unit}. Mask: {unit_masks[4]}"
                    )
                    continue
                _a[4] = a_resource_type
            action["per_position"][pos_idx, 1:] = _a
    return action
