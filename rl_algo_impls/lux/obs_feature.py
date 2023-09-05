from enum import IntEnum, auto

from rl_algo_impls.lux.actions import UNIT_ACTION_ENCODED_SIZE

observation_feature_length = 0


MULTI_DIM_FEATURES = dict(
    ICE_UNIT=4,
    ORE_UNIT=4,
    WATER_UNIT=4,
    METAL_UNIT=4,
    POWER_UNIT=4,
    ENQUEUED_ACTION=UNIT_ACTION_ENCODED_SIZE,
)


class _ObservationFeature(IntEnum):
    def _generate_next_value_(name: str, *args):
        global observation_feature_length
        sz = MULTI_DIM_FEATURES.get(name, 1)
        observation_feature_length += sz
        return observation_feature_length - sz


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
