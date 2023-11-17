from collections import defaultdict
from enum import IntEnum, auto

from rl_algo_impls.lux.actions import (
    SIMPLE_UNIT_ACTION_ENCODED_SIZE,
    UNIT_ACTION_ENCODED_SIZE,
)

MULTI_DIM_FEATURES = dict(
    ICE_UNIT=4,
    ORE_UNIT=4,
    WATER_UNIT=4,
    METAL_UNIT=4,
    POWER_UNIT=4,
    ENQUEUED_ACTION=UNIT_ACTION_ENCODED_SIZE,
)

OBSERVATION_FEATURE_LENGTH = 0


class _ObservationFeature(IntEnum):
    def _generate_next_value_(name: str, *args):
        global OBSERVATION_FEATURE_LENGTH
        sz = MULTI_DIM_FEATURES.get(name, 1)
        OBSERVATION_FEATURE_LENGTH += sz
        return OBSERVATION_FEATURE_LENGTH - sz


class ObservationFeature(_ObservationFeature):
    X = auto()
    Y = auto()
    ORE = auto()
    ICE = auto()
    NON_ZERO_RUBBLE = auto()
    RUBBLE = auto()
    NON_ZERO_LICHEN = auto()
    LICHEN = auto()
    SPREADABLE_LICHEN = auto()  # Consider removing
    OWN_LICHEN = auto()
    OPPONENT_LICHEN = auto()
    FACTORY = auto()  # Consider removing
    OWN_FACTORY = auto()
    OPPONENT_FACTORY = auto()
    ICE_FACTORY = auto()
    WATER_FACTORY = auto()
    ORE_FACTORY = auto()
    METAL_FACTORY = auto()
    POWER_FACTORY = auto()
    CAN_BUILD_LIGHT_ROBOT = auto()  # Consider removing
    CAN_BUILD_HEAVY_ROBOT = auto()  # Consider removing
    CAN_WATER_LICHEN = auto()  # Consider removing
    DAY_SURVIVE_FACTORY = auto()  # Consider removing
    OVER_DAY_SURVIVE_FACTORY = auto()  # Consider removing
    DAY_SURVIVE_WATER_FACTORY = auto()  # Consider removing
    CONNECTED_LICHEN_TILES = auto()  # Consider removing
    CONNECTED_LICHEN = auto()  # Consider removing
    IS_FACTORY_TILE = auto()
    IS_OWN_FACTORY_TILE = auto()
    IS_OPPONENT_FACTORY_TILE = auto()
    UNIT = auto()  # Consider removing
    OWN_UNIT = auto()
    OPPONENT_UNIT = auto()
    UNIT_IS_HEAVY = auto()
    ICE_UNIT = auto()
    ORE_UNIT = auto()
    WATER_UNIT = auto()
    METAL_UNIT = auto()
    POWER_UNIT = auto()
    ENQUEUED_ACTION = auto()
    CAN_COLLIDE_WITH_FRIENDLY_UNIT = auto()  # Consider removing
    GAME_PROGRESS = auto()
    DAY_CYCLE = auto()
    FACTORIES_TO_PLACE = auto()


SIMPLE_MULTI_DIM_FEATURES = dict(
    ICE_UNIT=4,
    ORE_UNIT=4,
    WATER_UNIT=4,
    METAL_UNIT=4,
    POWER_UNIT=4,
    ENQUEUED_ACTION=SIMPLE_UNIT_ACTION_ENCODED_SIZE,
    OWN_UNIT_COULD_BE_IN_DIRECTION=4,
)


SIMPLE_OBSERVATION_FEATURE_LENGTH = 0

ICE_FACTORY_LAMBDA = 1 / 500
ORE_FACTORY_LAMBDA = 1 / 1000
WATER_FACTORY_LAMBDA = 1 / 1000
METAL_FACTORY_LAMBDA = 1 / 100
POWER_FACTORY_LAMBDA = 1 / 3000
ICE_WATER_FACTORY_LAMBDA = 1 / 50
WATER_COST_LAMBDA = 1 / 10

FACTORY_RESOURCE_LAMBDAS = (
    ICE_FACTORY_LAMBDA,
    ORE_FACTORY_LAMBDA,
    WATER_FACTORY_LAMBDA,
    METAL_FACTORY_LAMBDA,
    POWER_FACTORY_LAMBDA,
)

MIN_WATER_TO_PICKUP = 150


class _SimpleObservationFeature(IntEnum):
    def _generate_next_value_(name: str, *args):
        global SIMPLE_OBSERVATION_FEATURE_LENGTH
        sz = SIMPLE_MULTI_DIM_FEATURES.get(name, 1)
        SIMPLE_OBSERVATION_FEATURE_LENGTH += sz
        return SIMPLE_OBSERVATION_FEATURE_LENGTH - sz


class SimpleObservationFeature(_SimpleObservationFeature):
    X = auto()
    Y = auto()
    ICE = auto()
    ORE = auto()
    NON_ZERO_RUBBLE = auto()
    RUBBLE = auto()
    LICHEN = auto()
    LICHEN_AT_ONE = auto()
    OWN_LICHEN = auto()
    OPPONENT_LICHEN = auto()
    GAME_PROGRESS = auto()
    DAY_CYCLE = auto()
    FACTORIES_TO_PLACE = auto()
    OWN_FACTORY = auto()
    OPPONENT_FACTORY = auto()
    ICE_WATER_FACTORY = auto()
    WATER_COST = auto()
    IS_OWN_FACTORY_TILE = auto()
    IS_OPPONENT_FACTORY_TILE = auto()
    OWN_UNIT = auto()
    OPPONENT_UNIT = auto()
    UNIT_IS_HEAVY = auto()
    ICE_FACTORY = auto()
    ORE_FACTORY = auto()
    WATER_FACTORY = auto()
    METAL_FACTORY = auto()
    POWER_FACTORY = auto()
    ICE_UNIT = auto()
    ORE_UNIT = auto()
    WATER_UNIT = auto()
    METAL_UNIT = auto()
    POWER_UNIT = auto()
    ENQUEUED_ACTION = auto()
    OWN_UNIT_COULD_BE_IN_DIRECTION = auto()
