import logging
from dataclasses import astuple
from functools import total_ordering
from typing import Any, Dict, List, NamedTuple, Optional, TypeVar, Union

import numpy as np
from luxai_s2.actions import move_deltas

from rl_algo_impls.lux.shared import (
    LuxEnvConfig,
    LuxFactory,
    LuxGameState,
    LuxUnit,
    move_power_cost,
    pos_to_idx,
    pos_to_numpy,
)
from rl_algo_impls.lux.stats import ActionStats

FACTORY_ACTION_SIZES = (
    4,  # build light robot, build heavy robot, water lichen, do nothing
)
FACTORY_ACTION_ENCODED_SIZE = sum(FACTORY_ACTION_SIZES)

FACTORY_DO_NOTHING_ACTION = 3


UNIT_ACTION_SIZES = (
    6,  # action type
    5,  # move direction
    5,  # transfer direction
    5,  # transfer resource
    5,  # pickup resource
)
UNIT_ACTION_ENCODED_SIZE = sum(UNIT_ACTION_SIZES)


ACTION_SIZES = FACTORY_ACTION_SIZES + UNIT_ACTION_SIZES


def to_lux_actions(
    player: str,
    state: LuxGameState,
    actions: np.ndarray,
    action_mask: np.ndarray,
    enqueued_actions: Dict[str, Optional[np.ndarray]],
    action_stats: ActionStats,
) -> Dict[str, Any]:
    cfg = state.env_cfg

    lux_actions = {}

    positions_occupied: Dict[int, str] = {}

    def cancel_action(unit: LuxUnit):
        enqueued_action = enqueued_actions.get(unit.unit_id)
        if enqueued_action is not None and enqueued_action[0] != 5:
            lux_actions[unit.unit_id] = [
                np.array([5, 0, 0, u.battery_capacity, 0, cfg.max_episode_length])
            ]
        elif unit.unit_id in lux_actions:
            del lux_actions[unit.unit_id]

    def cancel_move(unit: LuxUnit, target_pos_idx: int):
        cancel_action(unit)

        action_stats.move_cancelled += 1

        if positions_occupied.get(target_pos_idx) == unit.unit_id:
            del positions_occupied[target_pos_idx]
        current_pos_idx = pos_to_idx(u.pos, cfg.map_size)
        if current_pos_idx in positions_occupied:
            cancel_move(
                state.units[player][positions_occupied[current_pos_idx]],
                current_pos_idx,
            )
        positions_occupied[current_pos_idx] = unit.unit_id

    unit_actions = []
    for u in state.units[player].values():
        if no_valid_unit_actions(u, action_mask, cfg.map_size):
            if cfg.verbose > 1:
                logging.warn(f"{state.real_env_steps}: No valid action for unit {u}")
            action_stats.no_valid_action += 1
            positions_occupied[pos_to_idx(u.pos, cfg.map_size)] = u.unit_id
            continue
        unit_actions.append(UnitAction(u, actions[pos_to_idx(u.pos, cfg.map_size), 1:]))
    unit_actions = sorted(unit_actions)

    for u, a in unit_actions:
        action_stats.action_type[a[0]] += 1

        def resource_amount(unit: Union[LuxUnit, LuxFactory], idx: int) -> int:
            if idx == 4:
                return unit.power
            return astuple(unit.cargo)[idx]

        if a[0] == 0:  # move
            direction = a[1]
            target_pos_idx = pos_to_idx(
                pos_to_numpy(u.pos) + move_deltas[direction], cfg.map_size
            )
            if target_pos_idx in positions_occupied:
                cancel_move(u, target_pos_idx)
                continue
            positions_occupied[target_pos_idx] = u.unit_id
            resource = 0
            amount = 0
            repeat = max_move_repeats(u, direction, state)
        else:
            positions_occupied[pos_to_idx(u.pos, cfg.map_size)] = u.unit_id

            if a[0] == 1:  # transfer
                direction = a[2]
                resource = a[3]
                amount = resource_amount(
                    u, resource
                )  # TODO: This can lead to waste (especially for light robots)
                repeat = cfg.max_episode_length
            elif a[0] == 2:  # pickup
                direction = 0
                resource = a[4]
                _capacity = u.cargo_space if resource < 4 else u.battery_capacity
                _factory = factory_at_pos(state, pos_to_numpy(u.pos))
                assert _factory is not None
                _factory_amount = resource_amount(_factory, resource)
                amount = max(
                    min(
                        _capacity - resource_amount(u, resource),
                        _factory_amount - int(min_factory_resources(cfg)[resource]),
                    ),
                    0,
                )
                assert amount > 0
                repeat = 1
            elif a[0] == 3:  # dig
                direction = 0
                resource = 0
                amount = 0
                repeat = u.power // u.unit_cfg.DIG_COST
            elif a[0] == 4:  # self-destruct
                direction = 0
                resource = 0
                amount = 0
                repeat = 1
            elif a[0] == 5:  # recharge
                direction = 0
                resource = 0
                amount = u.battery_capacity
                repeat = 1
            else:
                raise ValueError(f"Unrecognized action {a[0]}")
        assert repeat > 0
        if actions_equal(a, enqueued_actions.get(u.unit_id)):
            action_stats.repeat_action += 1
            continue

        assert u.power >= u.unit_cfg.ACTION_QUEUE_POWER_COST
        lux_actions[u.unit_id] = [
            np.array([a[0], direction, resource, amount, 0, repeat])
        ]

    # Check transfers have valid targets
    for u, a in unit_actions:
        if a[0] != 1:
            continue
        target_pos = pos_to_numpy(u.pos) + move_deltas[a[2]]
        if state.board.factory_occupancy_map[target_pos[0], target_pos[1]] != -1:
            continue
        target_pos_idx = pos_to_idx(target_pos, cfg.map_size)
        if target_pos_idx not in positions_occupied:
            cancel_action(u)
            action_stats.transfer_cancelled += 1

    for f in state.factories[player].values():
        a = actions[pos_to_idx(f.pos, cfg.map_size), 0]
        if a != FACTORY_DO_NOTHING_ACTION:
            if a in {0, 1} and pos_to_idx(f.pos, cfg.map_size) in positions_occupied:
                action_stats.build_cancelled += 1
                continue
            lux_actions[f.unit_id] = a
    return lux_actions


UnitActionSelf = TypeVar("UnitActionSelf", bound="UnitAction")


def is_move_action(action: np.ndarray) -> bool:
    return action[0] == 0 and action[1] > 0


@total_ordering
class UnitAction(NamedTuple):
    unit: LuxUnit
    action: np.ndarray

    def __lt__(self: UnitActionSelf, other: UnitActionSelf) -> bool:
        # Units that aren't moving have priority
        is_move = is_move_action(self.action)
        is_other_move = is_move_action(other.action)
        if is_move != is_other_move:
            return not is_move

        # Next heavy units have priority
        is_unit_heavy = self.unit.is_heavy()
        is_other_unit_heavy = other.unit.is_heavy()
        if is_unit_heavy != is_other_unit_heavy:
            return is_unit_heavy

        return False

    def __eq__(self: UnitActionSelf, other: UnitActionSelf) -> bool:
        is_move = is_move_action(self.action)
        is_other_move = is_move_action(other.action)
        is_unit_heavy = self.unit.is_heavy()
        is_other_unit_heavy = other.unit.is_heavy()
        return is_move == is_other_move and is_unit_heavy == is_other_unit_heavy


def is_position_in_map(pos: np.ndarray, config: LuxEnvConfig) -> bool:
    return bool(np.all(pos >= 0) and np.all(pos < config.map_size))


def max_move_repeats(unit: LuxUnit, direction_idx: int, state: LuxGameState) -> int:
    config = state.env_cfg
    num_repeats = 0
    power_remaining = unit.power
    move_delta = move_deltas[direction_idx]
    assert not bool(np.all(move_delta == 0)), "No move Move action not expected"
    target_pos = pos_to_numpy(unit.pos)
    while True:
        target_pos = target_pos + move_delta
        if not is_position_in_map(target_pos, config):
            return num_repeats
        rubble = int(state.board.rubble[target_pos[0], target_pos[1]])
        power_remaining -= move_power_cost(unit, rubble)
        if power_remaining < 0:
            return num_repeats
        num_repeats += 1


def enqueued_action_from_obs(action_queue: List[np.ndarray]) -> Optional[np.ndarray]:
    if len(action_queue) == 0:
        return None
    action = action_queue[0]
    action_type = action[0]
    if action_type == 0:
        return np.array((action_type, action[1], -1, -1, -1))
    elif action_type == 1:
        return np.array((action_type, -1, action[1], action[2], -1))
    elif action_type == 2:
        return np.array((action_type, -1, -1, -1, action[2]))
    elif 3 <= action_type <= 5:
        return np.array((action_type, -1, -1, -1, -1))
    else:
        raise ValueError(f"action_type {action_type} not supported: {action}")


def actions_equal(action: np.ndarray, enqueued: Optional[np.ndarray]) -> bool:
    if enqueued is None:
        return False
    return bool(np.all(np.where(enqueued == -1, True, action == enqueued)))


def no_valid_unit_actions(
    unit: LuxUnit, action_mask: np.ndarray, map_size: int
) -> bool:
    return not np.any(
        action_mask[
            pos_to_idx(unit.pos, map_size),
            FACTORY_ACTION_ENCODED_SIZE : FACTORY_ACTION_ENCODED_SIZE + 6,
        ]
    )


def factory_at_pos(state: LuxGameState, pos: np.ndarray) -> Optional[LuxFactory]:
    factory_idx = state.board.factory_occupancy_map[pos[0], pos[1]]
    if factory_idx == -1:
        return None
    factory_id = f"factory_{factory_idx}"
    if factory_id in state.factories["player_0"]:
        return state.factories["player_0"][factory_id]
    else:
        return state.factories["player_1"][factory_id]


def min_factory_resources(cfg: LuxEnvConfig) -> np.ndarray:
    return np.array([0, 0, cfg.FACTORY_WATER_CONSUMPTION * cfg.CYCLE_LENGTH, 0, 0])
