from typing import List, Optional, Union

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
from luxai_s2.map.position import Position
from luxai_s2.unit import Unit

from rl_algo_impls.shared.lux.shared import LuxEnvConfig, pos_to_numpy

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


def pos_to_idx(pos: Union[Position, np.ndarray], map_size: int) -> int:
    pos = pos_to_numpy(pos)
    return pos[0] * map_size + pos[1]


def max_move_repeats(unit: Unit, direction_idx: int, config: LuxEnvConfig) -> int:
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
