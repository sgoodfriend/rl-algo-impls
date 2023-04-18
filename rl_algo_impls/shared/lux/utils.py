from luxai_s2.config import EnvConfig
from luxai_s2.factory import Factory


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
    return factory.cargo.water >= factory.water_cost(config)
