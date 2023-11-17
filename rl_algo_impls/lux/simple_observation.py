from typing import Dict, Optional

import numpy as np
from luxai_s2.factory import FactoryStateDict
from luxai_s2.state import ObservationStateDict
from luxai_s2.unit import UnitStateDict

from rl_algo_impls.lux.actions import SIMPLE_UNIT_ACTION_SIZES
from rl_algo_impls.lux.obs_feature import (
    ICE_FACTORY_LAMBDA,
    ICE_WATER_FACTORY_LAMBDA,
    METAL_FACTORY_LAMBDA,
    ORE_FACTORY_LAMBDA,
    POWER_FACTORY_LAMBDA,
    SIMPLE_OBSERVATION_FEATURE_LENGTH,
    WATER_COST_LAMBDA,
    WATER_FACTORY_LAMBDA,
    SimpleObservationFeature,
)
from rl_algo_impls.lux.shared import LuxGameState, factory_water_cost

VERIFY = False


def simple_obs_from_lux_observation(
    player: str,
    lux_obs: ObservationStateDict,
    state: LuxGameState,
    enqueued_actions: Dict[str, Optional[np.ndarray]],
    other_own_could_be_at_destination: np.ndarray,
) -> np.ndarray:
    env_cfg = state.env_cfg
    map_size = env_cfg.map_size
    LIGHT_ROBOT_CFG = env_cfg.ROBOTS["LIGHT"]
    HEAVY_ROBOT_CFG = env_cfg.ROBOTS["HEAVY"]

    p1 = player
    p2 = [p for p in state.teams if p != player][0]

    obs = np.zeros(
        (SIMPLE_OBSERVATION_FEATURE_LENGTH, map_size, map_size), dtype=np.float32
    )
    obs[SimpleObservationFeature.X] = np.transpose(
        np.tile(np.linspace(-1, 1, num=map_size), (map_size, 1))
    )
    obs[SimpleObservationFeature.Y] = np.tile(
        np.linspace(-1, 1, num=map_size), (map_size, 1)
    )

    obs[SimpleObservationFeature.ORE] = lux_obs["board"]["ore"]
    obs[SimpleObservationFeature.ICE] = lux_obs["board"]["ice"]

    _rubble = lux_obs["board"]["rubble"]
    obs[SimpleObservationFeature.NON_ZERO_RUBBLE] = _rubble > 0
    obs[SimpleObservationFeature.RUBBLE] = _rubble / env_cfg.MAX_RUBBLE

    _lichen = lux_obs["board"]["lichen"]
    obs[SimpleObservationFeature.LICHEN] = _lichen / env_cfg.MAX_LICHEN_PER_TILE
    obs[SimpleObservationFeature.LICHEN_AT_ONE] = _lichen == 1

    _non_zero_lichen = _lichen > 0
    _lichen_strains = lux_obs["board"]["lichen_strains"]
    _own_lichen_strains = state.teams[p1].factory_strains
    _opponent_lichen_strains = state.teams[p2].factory_strains
    _own_lichen = np.isin(_lichen_strains, _own_lichen_strains)
    obs[SimpleObservationFeature.OWN_LICHEN] = _own_lichen
    obs[SimpleObservationFeature.OPPONENT_LICHEN] = _non_zero_lichen & ~_own_lichen

    obs[SimpleObservationFeature.GAME_PROGRESS] = (
        lux_obs["real_env_steps"] / env_cfg.max_episode_length
    )
    _day_idx = np.maximum(lux_obs["real_env_steps"], 0) % env_cfg.CYCLE_LENGTH
    obs[SimpleObservationFeature.DAY_CYCLE] = 1 - _day_idx / env_cfg.DAY_LENGTH

    obs[SimpleObservationFeature.FACTORIES_TO_PLACE] = (
        lux_obs["teams"][p1]["factories_to_place"] / env_cfg.MAX_FACTORIES
    )

    def add_factory(f: FactoryStateDict, p_id: str, is_own: bool) -> None:
        f_state = state.factories[p_id][f["unit_id"]]
        x, y = f["pos"]
        if VERIFY:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    assert (
                        state.board.factory_occupancy_map[x + dx, y + dy]
                        == f["strain_id"]
                    )
        if is_own:
            obs[SimpleObservationFeature.OWN_FACTORY, x, y] = True
        else:
            obs[SimpleObservationFeature.OPPONENT_FACTORY, x, y] = True

        _cargo = f["cargo"]
        _ice = _cargo["ice"]
        _water = _cargo["water"]
        obs[SimpleObservationFeature.ICE_FACTORY, x, y] = 1 - np.exp(
            -ICE_FACTORY_LAMBDA * _ice
        )
        obs[SimpleObservationFeature.ORE_FACTORY, x, y] = 1 - np.exp(
            -ORE_FACTORY_LAMBDA * _cargo["ore"]
        )
        obs[SimpleObservationFeature.WATER_FACTORY, x, y] = 1 - np.exp(
            -WATER_FACTORY_LAMBDA * _water
        )
        obs[SimpleObservationFeature.METAL_FACTORY, x, y] = 1 - np.exp(
            -METAL_FACTORY_LAMBDA * _cargo["metal"]
        )
        obs[SimpleObservationFeature.POWER_FACTORY, x, y] = 1 - np.exp(
            -POWER_FACTORY_LAMBDA * f["power"]
        )
        _ice_water_factory = _water + np.floor(_ice / env_cfg.ICE_WATER_RATIO)
        obs[SimpleObservationFeature.ICE_WATER_FACTORY, x, y] = 1 - np.exp(
            -ICE_WATER_FACTORY_LAMBDA * _ice_water_factory
        )

        grow_lichen_positions = f_state.grow_lichen_positions
        water_cost = 1 - np.exp(
            -WATER_COST_LAMBDA * factory_water_cost(f_state, env_cfg)
        )
        for gl_x, gl_y in grow_lichen_positions:
            obs[SimpleObservationFeature.WATER_COST, gl_x, gl_y] = water_cost

    for f in lux_obs["factories"][p1].values():
        add_factory(f, p1, True)
    for f in lux_obs["factories"][p2].values():
        add_factory(f, p2, False)

    _factory_cargo = obs[
        SimpleObservationFeature.ICE_FACTORY : SimpleObservationFeature.POWER_FACTORY
        + 1
    ]
    _factory_cargo = (
        _factory_cargo
        + np.roll(_factory_cargo, shift=-1, axis=-1)
        + np.roll(_factory_cargo, shift=1, axis=-1)
    )
    obs[
        SimpleObservationFeature.ICE_FACTORY : SimpleObservationFeature.POWER_FACTORY
        + 1
    ] = (
        _factory_cargo
        + np.roll(_factory_cargo, shift=-1, axis=-2)
        + np.roll(_factory_cargo, shift=1, axis=-2)
    )

    _ice_water_factory = obs[SimpleObservationFeature.ICE_WATER_FACTORY]
    _ice_water_factory = (
        _ice_water_factory
        + np.roll(_ice_water_factory, shift=-1, axis=-1)
        + np.roll(_ice_water_factory, shift=1, axis=-1)
    )
    obs[SimpleObservationFeature.ICE_WATER_FACTORY] = (
        _ice_water_factory
        + np.roll(_ice_water_factory, shift=-1, axis=-2)
        + np.roll(_ice_water_factory, shift=1, axis=-2)
    )

    obs[SimpleObservationFeature.IS_OWN_FACTORY_TILE] = np.isin(
        state.board.factory_occupancy_map, _own_lichen_strains
    )
    obs[SimpleObservationFeature.IS_OPPONENT_FACTORY_TILE] = np.isin(
        state.board.factory_occupancy_map, _opponent_lichen_strains
    )

    def add_unit(u: UnitStateDict, is_own: bool) -> None:
        x, y = u["pos"]
        if is_own:
            obs[SimpleObservationFeature.OWN_UNIT, x, y] = True
        else:
            obs[SimpleObservationFeature.OPPONENT_UNIT, x, y] = True
        _is_heavy = u["unit_type"] == "HEAVY"
        obs[SimpleObservationFeature.UNIT_IS_HEAVY, x, y] = _is_heavy

        _cargo_space = (
            HEAVY_ROBOT_CFG.CARGO_SPACE if _is_heavy else LIGHT_ROBOT_CFG.CARGO_SPACE
        )

        def add_cargo(v: int, idx: int) -> None:
            obs[idx : idx + 4, x, y] = (
                v / HEAVY_ROBOT_CFG.CARGO_SPACE,
                v / _cargo_space,
                v >= LIGHT_ROBOT_CFG.CARGO_SPACE,
                v == _cargo_space,
            )

        _cargo = u["cargo"]
        add_cargo(_cargo["ice"], SimpleObservationFeature.ICE_UNIT)
        add_cargo(_cargo["ore"], SimpleObservationFeature.ORE_UNIT)
        add_cargo(_cargo["water"], SimpleObservationFeature.WATER_UNIT)
        add_cargo(_cargo["metal"], SimpleObservationFeature.METAL_UNIT)

        _power = u["power"]
        _bat_cap = (
            HEAVY_ROBOT_CFG.BATTERY_CAPACITY
            if _is_heavy
            else LIGHT_ROBOT_CFG.BATTERY_CAPACITY
        )
        obs[
            SimpleObservationFeature.POWER_UNIT : SimpleObservationFeature.POWER_UNIT
            + 4,
            x,
            y,
        ] = (
            _power / HEAVY_ROBOT_CFG.BATTERY_CAPACITY,
            _power / _bat_cap,
            _power >= LIGHT_ROBOT_CFG.BATTERY_CAPACITY,
            _power == _bat_cap,
        )
        _enqueued_action = enqueued_actions.get(u["unit_id"])
        if _enqueued_action is not None:
            a_idx = 0
            for a, a_sz in zip(_enqueued_action, SIMPLE_UNIT_ACTION_SIZES):
                if a >= 0:
                    obs[SimpleObservationFeature.ENQUEUED_ACTION + a_idx + a, x, y] = 1
                a_idx += a_sz

    for u in lux_obs["units"][p1].values():
        add_unit(u, True)
    for u in lux_obs["units"][p2].values():
        add_unit(u, False)

    obs[
        SimpleObservationFeature.OWN_UNIT_COULD_BE_IN_DIRECTION : SimpleObservationFeature.OWN_UNIT_COULD_BE_IN_DIRECTION
        + 4
    ] = other_own_could_be_at_destination

    return obs
