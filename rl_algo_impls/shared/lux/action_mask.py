from dataclasses import astuple
from typing import Any, Dict, Optional, Tuple

import numpy as np
from luxai_s2.actions import move_deltas

from rl_algo_impls.shared.lux.actions import FACTORY_ACTION_ENCODED_SIZE, pos_to_idx
from rl_algo_impls.shared.lux.shared import (
    LuxEnvConfig,
    LuxFactory,
    LuxGameState,
    LuxUnit,
    agent_id,
    factory_water_cost,
    move_power_cost,
    pos_to_numpy,
)


def get_action_mask(
    player: str,
    state: LuxGameState,
    action_mask_shape: Tuple[int, int],
    enqueued_actions: Dict[str, Optional[np.ndarray]],
    move_masks: Dict[str, Any],
    move_validity_map: np.ndarray,
) -> np.ndarray:
    action_mask = np.full(
        action_mask_shape,
        False,
        dtype=np.bool_,
    )
    config = state.env_cfg
    for f in state.factories[player].values():
        action_mask[
            pos_to_idx(f.pos, config.map_size), :FACTORY_ACTION_ENCODED_SIZE
        ] = np.array(
            [
                is_build_light_valid(f, config),
                is_build_heavy_valid(f, config),
                is_water_action_valid(f, state, config),
                True,  # Do nothing is always valid
            ]
        )
    for u_id, u in state.units[player].items():
        enqueued_action = enqueued_actions.get(u_id)
        move_mask = move_masks[u_id]
        transfer_direction_mask = valid_transfer_direction_mask(
            u, state, config, move_mask, move_validity_map, enqueued_action
        )
        transfer_resource_mask = (
            valid_transfer_resource_mask(u)
            if np.any(transfer_direction_mask)
            else np.zeros(5)
        )
        pickup_resource_mask = valid_pickup_resource_mask(u, state, enqueued_action)
        valid_action_types = np.array(
            [
                np.any(move_mask),
                np.any(transfer_direction_mask),
                np.any(pickup_resource_mask),
                is_dig_valid(u, state, enqueued_action),
                if_self_destruct_valid(u, state, enqueued_action),
                is_recharge_valid(u, enqueued_action),
            ]
        )
        action_mask[
            pos_to_idx(u.pos, config.map_size), FACTORY_ACTION_ENCODED_SIZE:
        ] = np.concatenate(
            [
                valid_action_types,
                move_mask,
                transfer_direction_mask,
                transfer_resource_mask,
                pickup_resource_mask,
            ]
        )
    return action_mask


# Factory validity checks
def is_build_light_valid(factory: LuxFactory, config: LuxEnvConfig) -> bool:
    LIGHT_ROBOT = config.ROBOTS["LIGHT"]
    return (
        factory.cargo.metal >= LIGHT_ROBOT.METAL_COST
        and factory.power >= LIGHT_ROBOT.POWER_COST
    )


def is_build_heavy_valid(factory: LuxFactory, config: LuxEnvConfig) -> bool:
    HEAVY_ROBOT = config.ROBOTS["HEAVY"]
    return (
        factory.cargo.metal >= HEAVY_ROBOT.METAL_COST
        and factory.power >= HEAVY_ROBOT.POWER_COST
    )


def is_water_action_valid(
    factory: LuxFactory, state: LuxGameState, config: LuxEnvConfig
) -> bool:
    water_cost = factory_water_cost(factory, state, config)
    return factory.cargo.water >= water_cost


# Unit validity checks


def agent_move_masks(
    state: LuxGameState, player: str, enqueued_actions: Dict[str, Optional[np.ndarray]]
) -> Dict[str, np.ndarray]:
    return {
        u_id: valid_move_mask(u, state, enqueued_actions.get(u_id))
        for u_id, u in state.units[player].items()
    }


def valid_destination_map(
    state: LuxGameState, player: str, agent_move_masks: Dict[str, np.ndarray]
) -> np.ndarray:
    map_size = state.env_cfg.map_size
    move_validity_map = np.zeros((map_size, map_size), dtype=np.int8)
    for u_id, valid_moves_mask in agent_move_masks.items():
        u = state.units[player][u_id]
        pos = pos_to_numpy(u.pos)
        for direction_idx, move_delta in enumerate(move_deltas):
            if valid_moves_mask[direction_idx] or direction_idx == 0:
                move_validity_map[pos[0] + move_delta[0], pos[1] + move_delta[1]] += 1
    return move_validity_map


def is_position_in_map(pos: np.ndarray, config: LuxEnvConfig) -> bool:
    return bool(np.all(pos >= 0) and np.all(pos < config.map_size))


def valid_move_mask(
    unit: LuxUnit,
    state: LuxGameState,
    enqueued_action: Optional[np.ndarray],
) -> np.ndarray:
    config = state.env_cfg

    def is_valid_target(pos: np.ndarray, move_direction: int) -> bool:
        if not is_position_in_map(pos, config):
            return False
        factory_num_id = state.board.factory_occupancy_map[pos[0], pos[1]]
        if (
            factory_num_id != -1
            and f"factory_{factory_num_id}" not in state.factories[agent_id(unit)]
        ):
            return False
        rubble = int(state.board.rubble[pos[0], pos[1]])
        power_cost = move_power_cost(unit, rubble)
        if (
            enqueued_action is None
            or enqueued_action[0] != 0
            or enqueued_action[1] != move_direction
        ):
            power_cost += unit.unit_cfg.ACTION_QUEUE_POWER_COST
        if unit.power < power_cost:
            return False
        return True

    return np.array(
        [False]
        + [
            is_valid_target(pos_to_numpy(unit.pos) + move_delta, idx + 1)
            for idx, move_delta in enumerate(move_deltas[1:])
        ]
    )


def valid_transfer_direction_mask(
    unit: LuxUnit,
    state: LuxGameState,
    config: LuxEnvConfig,
    move_mask: np.ndarray,
    move_validity_map: np.ndarray,
    enqueued_action: Optional[np.ndarray],
) -> np.ndarray:
    if (
        enqueued_action is None or enqueued_action[0] != 1
    ) and unit.power < unit.unit_cfg.ACTION_QUEUE_POWER_COST:
        return np.full(5, False)

    def is_valid_target(pos: np.ndarray, move_direction: int) -> bool:
        if (
            enqueued_action is None or enqueued_action[2] != move_direction
        ) and unit.power < unit.unit_cfg.ACTION_QUEUE_POWER_COST:
            return False
        if not is_position_in_map(pos, config):
            return False
        factory_at_target = state.board.factory_occupancy_map[pos[0], pos[1]]
        if (
            factory_at_target != -1
            and f"factory_{factory_at_target}" in state.factories[agent_id(unit)]
        ):
            return True
        if move_direction == 0:
            # Center drop-off is just for factory
            return False
        return bool(
            move_validity_map[pos[0], pos[1]] - (1 if move_mask[move_direction] else 0)
            > 0
        )

    return np.array(
        [
            is_valid_target(pos_to_numpy(unit.pos) + move_delta, idx)
            for idx, move_delta in enumerate(move_deltas)
        ]
    )


def valid_transfer_resource_mask(unit: LuxUnit) -> np.ndarray:
    return np.array(astuple(unit.cargo) + (unit.power,)) > 0


def valid_pickup_resource_mask(
    unit: LuxUnit, state: LuxGameState, enqueued_action: Optional[np.ndarray]
) -> np.ndarray:
    has_power_to_change = unit.power >= unit.unit_cfg.ACTION_QUEUE_POWER_COST
    if (enqueued_action is None or enqueued_action[0] != 2) and not has_power_to_change:
        return np.zeros(5)
    pos = pos_to_numpy(unit.pos)
    factory_id = state.board.factory_occupancy_map[pos[0], pos[1]]
    if factory_id == -1:
        return np.zeros(5)
    factory = state.factories[agent_id(unit)][f"factory_{factory_id}"]
    has_resource = np.array(astuple(factory.cargo) + (factory.power,)) > 0
    has_capacity = np.concatenate(
        [
            np.array(astuple(unit.cargo)) < unit.cargo_space,
            np.array([unit.power < unit.battery_capacity]),
        ]
    )
    has_power = np.array(
        [
            has_power_to_change
            or (enqueued_action is not None and enqueued_action[4] == idx)
            for idx in range(5)
        ]
    )
    return has_resource * has_capacity * has_power


def is_dig_valid(
    unit: LuxUnit, state: LuxGameState, enqueued_action: Optional[np.ndarray]
) -> bool:
    power_cost = unit.unit_cfg.DIG_COST
    if enqueued_action is None or enqueued_action[0] != 3:
        power_cost += unit.unit_cfg.ACTION_QUEUE_POWER_COST
    if unit.power < power_cost:
        return False
    pos = pos_to_numpy(unit.pos)
    if (
        state.board.rubble[pos[0], pos[1]]
        or state.board.ice[pos[0], pos[1]]
        or state.board.ore[pos[0], pos[1]]
    ):
        return True
    lichen_strain = state.board.lichen_strains[pos[0], pos[1]]
    if (
        lichen_strain != -1
        and f"factory_{lichen_strain}" not in state.factories[agent_id(unit)]
    ):
        return True
    return False


def if_self_destruct_valid(
    unit: LuxUnit, state: LuxGameState, enqueued_action: Optional[np.ndarray]
) -> bool:
    pos = pos_to_numpy(unit.pos)
    factory_id = state.board.factory_occupancy_map[pos[0], pos[1]]
    if factory_id != -1:
        return False
    power_cost = unit.unit_cfg.SELF_DESTRUCT_COST
    if enqueued_action is None or enqueued_action[0] != 4:
        power_cost += unit.unit_cfg.ACTION_QUEUE_POWER_COST
    if unit.power < power_cost:
        return False
    return True


def is_recharge_valid(unit: LuxUnit, enqueued_action: Optional[np.ndarray]) -> bool:
    if (
        enqueued_action is None or enqueued_action[0] != 5
    ) and unit.power < unit.unit_cfg.ACTION_QUEUE_POWER_COST:
        return False
    return True
