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
    RECHARGE_UNIT_ACTION,
    UNIT_ACTION_SIZES,
)
from rl_algo_impls.lux.kit.kit import GameState
from rl_algo_impls.lux.kit.unit import Unit
from rl_algo_impls.lux.kit.utils import my_turn_to_place_factory
from rl_algo_impls.lux.observation import observation_and_action_mask
from rl_algo_impls.lux.rewards import MIN_SCORE, LuxRewardWeights
from rl_algo_impls.lux.shared import idx_to_pos, pos_to_idx
from rl_algo_impls.lux.vec_env.lux_replay_state import LuxReplayState, LuxState


class LuxReplayEnv(Env):
    def __init__(
        self,
        next_replay_path_fn: Callable[[], str],
        team_name: str,
        reward_weights: Optional[Dict[str, float]] = None,
        verify: bool = False,
        factory_ice_distance_buffer: Optional[int] = None,
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
        self.factory_ice_distance_buffer = factory_ice_distance_buffer

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
            factory_ice_distance_buffer=self.factory_ice_distance_buffer,
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
        self, action: Optional[Dict[str, np.ndarray]]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        state, replay_action, done, self_score, opponent_score = self.state.step()
        # _last_action must be set before _from_lux_state is called, which updates prior obs
        self._last_action = from_lux_action(
            self.action_space,
            state.player,
            replay_action,
            self._action_mask,
            self._last_game_state,
            self._last_enqueued_actions,
            action,
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
        self._last_enqueued_actions = enqueued_actions
        obs, self._action_mask = observation_and_action_mask(
            player,
            obs_state_dict,
            game_state,
            self.action_mask_shape,
            enqueued_actions,
            factory_ice_distance_buffer=self.factory_ice_distance_buffer,
        )
        return obs

    def get_action_mask(self) -> Dict[str, np.ndarray]:
        return self._action_mask

    def close(self) -> None:
        self.next_replay_path_fn = None

    @property
    def last_action(self) -> Dict[str, np.ndarray]:
        return self._last_action

    @property
    def max_episode_length(self) -> int:
        return self.state.env_cfg.max_episode_length


def from_lux_action(
    action_space: DictSpace,
    player: str,
    lux_action: Dict[str, Union[int, str, List[List[int]]]],
    action_mask: Dict[str, np.ndarray],
    game_state: GameState,
    enqueued_actions: Dict[str, Optional[np.ndarray]],
    fallback_action: Optional[Dict[str, np.ndarray]],
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
        spawn_action = pos_to_idx(pos, map_size)
        if not action_mask["pick_position"][0, spawn_action]:
            if fallback_action:
                spawn_action = fallback_action["pick_position"][0]
                logging.warn(
                    f"Invalid spawn action {pos}. Fallback: {idx_to_pos(spawn_action, map_size)}"
                )
            else:
                raise ValueError(f"Invalid spawn action {pos}. No fallback.")
        action["pick_position"][0] = spawn_action
        # TODO: Handle metal and water assignment
        return action
    # Bid or factory placement phase. No unit or factory actions.
    if game_state.real_env_steps < 0:
        assert not my_turn_to_place_factory(
            game_state.teams[player].place_first, game_state.env_steps
        )
        return action
    factory_actions = {}
    unit_actions = {}
    for u_id in game_state.units[player]:
        a = enqueued_actions.get(u_id, None)
        if a is not None:
            unit_actions[u_id] = np.where(a >= 0, a, 0)
        else:
            unit_actions[u_id] = None
    for fu_id, a in lux_action.items():
        if fu_id.startswith("factory"):
            assert isinstance(a, int)
            factory_actions[fu_id] = a
        else:
            assert isinstance(a, list)
            a_type, a_dir, a_resource_type, a_amount, a_repeat, a_n = a[0]
            _a = np.zeros(len(UNIT_ACTION_SIZES), dtype=np.int32)
            _a[0] = a_type
            if a_type == 0:
                _a[1] = a_dir
            elif a_type == 1:
                _a[2] = a_dir
                _a[3] = a_resource_type
            elif a_type == 2:
                _a[4] = a_resource_type
            unit_actions[fu_id] = _a
    for f_id, a in factory_actions.items():
        factory = game_state.factories[player][f_id]
        pos_idx = pos_to_idx(factory.pos, map_size)
        factory_mask = action_mask["per_position"][
            pos_idx, :FACTORY_ACTION_ENCODED_SIZE
        ]
        factory_action = a + 1
        if not factory_mask[factory_action]:
            if fallback_action:
                fa = fallback_action["per_position"][pos_idx, 0]
                action["per_position"][pos_idx, 0] = fa
            else:
                fa = None
            logging.warn(
                f"Invalid factory action {factory_action} for factory {factory}. Mask {factory_mask}. Fallback {fa}"
            )
            continue
        action["per_position"][pos_idx, 0] = factory_action
    for u_id, a in unit_actions.items():
        unit = game_state.units[player][u_id]
        pos_idx = pos_to_idx(unit.pos, map_size)
        pos_mask = action_mask["per_position"][pos_idx]
        if not np.any(pos_mask[FACTORY_ACTION_ENCODED_SIZE:]):
            if a is not None:
                logging.info(f"Attempt action {a} despite no valid actions {unit}")
            continue
        unit_action = a
        if not is_valid_action(unit_action, pos_mask, unit):
            unit_action = get_fallback_action(pos_mask, fallback_action, pos_idx)
            if a is not None or unit_action[0] != RECHARGE_UNIT_ACTION:
                logging.info(f"{unit} action fallback: {a} -> {unit_action}")
        action["per_position"][pos_idx, 1:] = unit_action
    return action


def get_fallback_action(
    pos_mask: np.ndarray, fallback_action: Optional[Dict[str, np.ndarray]], pos_idx: int
) -> np.ndarray:
    unit_masks = []
    m_idx = FACTORY_ACTION_ENCODED_SIZE
    for mask_sz in UNIT_ACTION_SIZES:
        unit_masks.append(pos_mask[m_idx : m_idx + mask_sz])
        m_idx += mask_sz
    assert unit_masks[0].any(), f"No valid action types: {unit_masks}"
    a = np.zeros(len(UNIT_ACTION_SIZES), dtype=np.int32)
    if unit_masks[0][RECHARGE_UNIT_ACTION]:  # Recharge
        a[0] = RECHARGE_UNIT_ACTION
        return a
    if fallback_action:
        return fallback_action["per_position"][pos_idx, 1:]
    for idx, m in enumerate(unit_masks[0]):
        if not m:
            continue
        a[0] = idx
        if idx == 0:
            a[1] = np.where(unit_masks[1])[0]
        elif idx == 1:
            a[2] = np.where(unit_masks[2])[0]
            a[3] = np.where(unit_masks[3])[0]
        elif idx == 2:
            a[4] = np.where(unit_masks[4])[0]
        return a
    assert False, "unit_masks[0] earlier asserted to be non-zero. Unreachable"


def is_valid_action(
    action: Optional[np.ndarray], pos_mask: np.ndarray, unit: Unit
) -> bool:
    if action is None:
        return False
    m_idx = FACTORY_ACTION_ENCODED_SIZE
    for idx, mask_sz in enumerate(UNIT_ACTION_SIZES):
        mask = pos_mask[m_idx : m_idx + mask_sz]
        if idx == 0:
            if not mask[action[idx]]:
                logging.info(
                    f"Invalid action type {action[idx]} for unit {unit}. Mask: {mask}"
                )
                return False
        elif idx == 1:
            if action[0] == 0 and not mask[action[idx]]:
                logging.info(
                    f"Invalid move dir {action[idx]} for unit {unit}. Mask: {mask}"
                )
                return False
        elif idx == 2:
            if action[0] == 1 and not mask[action[idx]]:
                logging.info(
                    f"Invalid transfer dir {action[idx]} for unit {unit}. Mask: {mask}"
                )
                return False
        elif idx == 3:
            if action[0] == 1 and not mask[action[idx]]:
                logging.info(
                    f"Invalid transfer resource {action[idx]} for unit {unit}. Mask: {mask}"
                )
                return False
        elif idx == 4:
            if action[0] == 2 and not mask[action[idx]]:
                logging.info(
                    f"Invalid pickup resource {action[idx]} for unit {unit}. Mask: {mask}"
                )
                return False
        m_idx += mask_sz
    return True
