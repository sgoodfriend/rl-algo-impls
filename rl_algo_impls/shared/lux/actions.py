from dataclasses import astuple
from typing import List, Optional

import numpy as np
from luxai_s2.actions import (
    Action,
    DigAction,
    MoveAction,
    PickupAction,
    RechargeAction,
    SelfDestructAction,
    TransferAction,
    move_deltas,
)
from luxai_s2.config import EnvConfig
from luxai_s2.factory import Factory
from luxai_s2.state import State
from luxai_s2.unit import Unit

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


def is_build_light_valid(factory: Factory, config: EnvConfig) -> bool:
    LIGHT_ROBOT = config.ROBOTS["LIGHT"]
    return (
        factory.cargo.metal >= LIGHT_ROBOT.METAL_COST
        and factory.power >= LIGHT_ROBOT.POWER_COST
    )


def is_build_heavy_valid(factory: Factory, config: EnvConfig) -> bool:
    HEAVY_ROBOT = config.ROBOTS["HEAVY"]
    return (
        factory.cargo.metal >= HEAVY_ROBOT.METAL_COST
        and factory.power >= HEAVY_ROBOT.POWER_COST
    )


def is_water_action_valid(factory: Factory, config: EnvConfig) -> bool:
    water_cost = factory.water_cost(config)
    return water_cost > 0 and factory.cargo.water >= factory.water_cost(config)


def max_move_repeats(unit: Unit, direction_idx: int, config: EnvConfig) -> int:
    def steps_til_edge(p: int, delta: int) -> int:
        if delta < 0:
            return p
        else:
            return config.map_size - p - 1

    move_delta = move_deltas[direction_idx]
    if move_delta[0]:
        return steps_til_edge(unit.pos.x, move_delta[0])
    else:
        return steps_til_edge(unit.pos.y, move_delta[1])


def is_position_in_map(pos: np.ndarray, config: EnvConfig) -> bool:
    return np.all(pos >= 0) and np.all(pos < config.map_size)


def valid_move_mask(
    unit: Unit, state: State, config: EnvConfig, enqueued_action: Optional[np.ndarray]
) -> np.ndarray:
    def is_valid_target(pos: np.ndarray, move_direction: int) -> bool:
        if not is_position_in_map(pos, config):
            return False
        factory_at_target = state.board.factory_occupancy_map[pos[0], pos[1]]
        if (
            factory_at_target != -1
            and f"factory_{factory_at_target}" not in state.factories[unit.team.agent]
        ):
            return False
        rubble = state.board.rubble[pos[0], pos[1]]
        power_cost = unit.move_power_cost(rubble)
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
            is_valid_target(unit.pos.pos + move_delta, idx + 1)
            for idx, move_delta in enumerate(move_deltas[1:])
        ]
    )


def valid_transfer_direction_mask(
    unit: Unit,
    state: State,
    config: EnvConfig,
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
            and f"factory_{factory_at_target}" in state.factories[unit.team.agent]
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
            is_valid_target(unit.pos.pos + move_delta, idx)
            for idx, move_delta in enumerate(move_deltas)
        ]
    )


def valid_transfer_resource_mask(unit: Unit) -> np.ndarray:
    return np.array(astuple(unit.cargo) + (unit.power,)) > 0


def valid_pickup_resource_mask(
    unit: Unit, state: State, enqueued_action: Optional[np.ndarray]
) -> np.ndarray:
    has_power_to_change = unit.power >= unit.unit_cfg.ACTION_QUEUE_POWER_COST
    if (enqueued_action is None or enqueued_action[0] != 2) and not has_power_to_change:
        return np.zeros(5)
    factory_id = state.board.factory_occupancy_map[unit.pos.x, unit.pos.y]
    if factory_id == -1:
        return np.zeros(5)
    factory = state.factories[unit.team.agent][f"factory_{factory_id}"]
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
    unit: Unit, state: State, enqueued_action: Optional[np.ndarray]
) -> bool:
    power_cost = unit.unit_cfg.DIG_COST
    if enqueued_action is None or enqueued_action[0] != 3:
        power_cost += unit.unit_cfg.ACTION_QUEUE_POWER_COST
    if unit.power < power_cost:
        return False
    if (
        state.board.rubble[unit.pos.x, unit.pos.y]
        or state.board.ice[unit.pos.x, unit.pos.y]
        or state.board.ore[unit.pos.x, unit.pos.y]
    ):
        return True
    lichen_strain = state.board.lichen_strains[unit.pos.x, unit.pos.y]
    if (
        lichen_strain != -1
        and f"factory_{lichen_strain}" not in state.factories[unit.team.agent]
    ):
        return True
    return False


def if_self_destruct_valid(
    unit: Unit, state: State, enqueued_action: Optional[np.ndarray]
) -> bool:
    factory_id = state.board.factory_occupancy_map[unit.pos.x, unit.pos.y]
    if factory_id != -1:
        return False
    power_cost = unit.unit_cfg.SELF_DESTRUCT_COST
    if enqueued_action is None or enqueued_action[0] != 4:
        power_cost += unit.unit_cfg.ACTION_QUEUE_POWER_COST
    if unit.power < power_cost:
        return False
    return True


def is_recharge_valid(unit: Unit, enqueued_action: Optional[np.ndarray]) -> bool:
    if (
        enqueued_action is None or enqueued_action[0] != 5
    ) and unit.power < unit.unit_cfg.ACTION_QUEUE_POWER_COST:
        return False
    return True


def action_array_from_queue(action_queue: List[Action]) -> Optional[np.ndarray]:
    if len(action_queue) == 0:
        return None
    action = action_queue[0]
    if isinstance(action, MoveAction):
        return np.array((0, action.move_dir, -1, -1, -1))
    elif isinstance(action, TransferAction):
        return np.array((1, -1, action.transfer_dir, action.resource, -1))
    elif isinstance(action, PickupAction):
        return np.array((2, -1, -1, -1, action.resource))
    elif isinstance(action, DigAction):
        return np.array((3, -1, -1, -1, -1))
    elif isinstance(action, SelfDestructAction):
        return np.array((4, -1, -1, -1, -1))
    elif isinstance(action, RechargeAction):
        return np.array((5, -1, -1, -1, -1))
    else:
        raise ValueError(f"{action.__class__.__name__} not supported")


def actions_equal(action: np.ndarray, enqueued: Optional[np.ndarray]) -> bool:
    if enqueued is None:
        return False
    return bool(np.all(np.where(enqueued == -1, True, action == enqueued)))
