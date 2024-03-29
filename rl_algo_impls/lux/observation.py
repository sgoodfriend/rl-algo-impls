from typing import Dict, NamedTuple, Optional, Tuple, Type

import numpy as np
from luxai_s2.factory import FactoryStateDict
from luxai_s2.state import ObservationStateDict
from luxai_s2.unit import UnitStateDict

from rl_algo_impls.lux.action_mask import (
    agent_move_masks,
    agent_simple_move_masks,
    get_action_mask,
    get_adjacent_rubble,
    get_simple_action_mask,
    is_build_heavy_valid,
    is_build_light_valid,
    other_own_could_be_at_destination_map,
    valid_destination_map,
)
from rl_algo_impls.lux.actions import UNIT_ACTION_ENCODED_SIZE, UNIT_ACTION_SIZES
from rl_algo_impls.lux.agent_config import LuxAgentConfig
from rl_algo_impls.lux.obs_feature import OBSERVATION_FEATURE_LENGTH, ObservationFeature
from rl_algo_impls.lux.shared import LuxGameState, factory_water_cost
from rl_algo_impls.lux.simple_observation import simple_obs_from_lux_observation

ICE_FACTORY_MAX = 100_000
ORE_FACTORY_MAX = 50_000
WATER_FACTORY_MAX = 25_000
METAL_FACTORY_MAX = 10_000
POWER_FACTORY_MAX = 50_000

LICHEN_TILES_FACTORY_MAX = 128
LICHEN_FACTORY_MAX = 128_000

VERIFY = False


class ObservationAndActionMask(NamedTuple):
    observation: np.ndarray
    action_mask: Dict[str, np.ndarray]


def observation_and_action_mask(
    player: str,
    lux_obs: ObservationStateDict,
    state: LuxGameState,
    action_mask_shape: Dict[str, Tuple[int, int]],
    enqueued_actions: Dict[str, Optional[np.ndarray]],
    agent_cfg: LuxAgentConfig,
) -> ObservationAndActionMask:
    adjacent_rubble = get_adjacent_rubble(state, player)
    if agent_cfg.use_simplified_spaces:
        move_masks = agent_simple_move_masks(
            state, player, enqueued_actions, adjacent_rubble
        )
        other_own_could_be_at_destination = other_own_could_be_at_destination_map(
            state, player, move_masks
        )
        action_mask = get_simple_action_mask(
            player,
            state,
            action_mask_shape,
            enqueued_actions,
            move_masks,
            other_own_could_be_at_destination,
            agent_cfg,
            adjacent_rubble,
        )
        observation = simple_obs_from_lux_observation(
            player,
            lux_obs,
            state,
            enqueued_actions,
            other_own_could_be_at_destination,
        )
    else:
        move_masks = agent_move_masks(state, player, enqueued_actions)
        move_validity_map = valid_destination_map(state, player, move_masks)
        action_mask = get_action_mask(
            player,
            state,
            action_mask_shape,
            enqueued_actions,
            move_masks,
            move_validity_map,
            agent_cfg,
            adjacent_rubble,
        )
        observation = _from_lux_observation(
            player, lux_obs, state, enqueued_actions, move_validity_map
        )
    return ObservationAndActionMask(observation, action_mask)


def _from_lux_observation(
    player: str,
    lux_obs: ObservationStateDict,
    state: LuxGameState,
    enqueued_actions: Dict[str, Optional[np.ndarray]],
    move_validity_map: np.ndarray,
) -> np.ndarray:
    env_cfg = state.env_cfg
    map_size = env_cfg.map_size
    LIGHT_ROBOT = env_cfg.ROBOTS["LIGHT"]
    HEAVY_ROBOT = env_cfg.ROBOTS["HEAVY"]
    WATER_CONSUMED_IN_DAY = env_cfg.FACTORY_WATER_CONSUMPTION * env_cfg.CYCLE_LENGTH

    p1 = player
    p2 = [p for p in state.teams if p != player][0]

    obs = np.zeros((OBSERVATION_FEATURE_LENGTH, map_size, map_size), dtype=np.float32)
    obs[ObservationFeature.X] = np.transpose(
        np.tile(np.linspace(-1, 1, num=map_size), (map_size, 1))
    )
    obs[ObservationFeature.Y] = np.tile(np.linspace(-1, 1, num=map_size), (map_size, 1))

    obs[ObservationFeature.ORE] = lux_obs["board"]["ore"]
    obs[ObservationFeature.ICE] = lux_obs["board"]["ice"]

    _rubble = lux_obs["board"]["rubble"]
    obs[ObservationFeature.NON_ZERO_RUBBLE] = _rubble > 0
    obs[ObservationFeature.RUBBLE] = _rubble / env_cfg.MAX_RUBBLE

    _lichen = lux_obs["board"]["lichen"]
    _non_zero_lichen = _lichen > 0
    obs[ObservationFeature.NON_ZERO_LICHEN] = _non_zero_lichen
    obs[ObservationFeature.LICHEN] = _lichen / env_cfg.MAX_LICHEN_PER_TILE
    obs[ObservationFeature.SPREADABLE_LICHEN] = _lichen >= env_cfg.MIN_LICHEN_TO_SPREAD

    _lichen_strains = lux_obs["board"]["lichen_strains"]
    _own_lichen_strains = state.teams[p1].factory_strains
    _opponent_lichen_strains = state.teams[p2].factory_strains
    _own_lichen = np.isin(_lichen_strains, _own_lichen_strains)
    obs[ObservationFeature.OWN_LICHEN] = _own_lichen
    obs[ObservationFeature.OPPONENT_LICHEN] = _non_zero_lichen & ~_own_lichen

    _lichen_counts = {
        k: v for k, v in zip(*np.unique(_lichen_strains, return_counts=True))
    }

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
        obs[ObservationFeature.FACTORY, x, y] = True
        if is_own:
            obs[ObservationFeature.OWN_FACTORY, x, y] = True
        else:
            obs[ObservationFeature.OPPONENT_FACTORY, x, y] = True

        _cargo = f["cargo"]
        _ice = _cargo["ice"]
        _water = _cargo["water"]
        obs[ObservationFeature.ICE_FACTORY, x, y] = _ice / ICE_FACTORY_MAX
        obs[ObservationFeature.WATER_FACTORY, x, y] = _water / WATER_FACTORY_MAX
        obs[ObservationFeature.ORE_FACTORY, x, y] = _cargo["ore"] / ORE_FACTORY_MAX
        obs[ObservationFeature.METAL_FACTORY, x, y] = (
            _cargo["metal"] / METAL_FACTORY_MAX
        )
        obs[ObservationFeature.POWER_FACTORY, x, y] = f["power"] / POWER_FACTORY_MAX

        obs[ObservationFeature.CAN_BUILD_LIGHT_ROBOT, x, y] = is_build_light_valid(
            f_state, env_cfg
        )
        obs[ObservationFeature.CAN_BUILD_HEAVY_ROBOT, x, y] = is_build_heavy_valid(
            f_state, env_cfg
        )
        _water_lichen_cost = factory_water_cost(f_state, state, env_cfg)
        obs[ObservationFeature.CAN_WATER_LICHEN, x, y] = _water > _water_lichen_cost
        _water_supply = _water + _ice / env_cfg.ICE_WATER_RATIO
        obs[ObservationFeature.DAY_SURVIVE_FACTORY, x, y] = max(
            _water_supply / WATER_CONSUMED_IN_DAY, 1
        )
        obs[ObservationFeature.OVER_DAY_SURVIVE_FACTORY, x, y] = (
            _water_supply > WATER_CONSUMED_IN_DAY
        )
        obs[ObservationFeature.DAY_SURVIVE_WATER_FACTORY, x, y] = (
            _water_supply - _water_lichen_cost > WATER_CONSUMED_IN_DAY
        )
        obs[ObservationFeature.CONNECTED_LICHEN_TILES, x, y] = (
            _lichen_counts.get(f["strain_id"], 0) / LICHEN_TILES_FACTORY_MAX
        )
        obs[ObservationFeature.CONNECTED_LICHEN, x, y] = (
            np.sum(np.where(_lichen_strains == f["strain_id"], _lichen, 0))
            / LICHEN_FACTORY_MAX
        )

    for f in lux_obs["factories"][p1].values():
        add_factory(f, p1, True)
    for f in lux_obs["factories"][p2].values():
        add_factory(f, p2, False)

    obs[ObservationFeature.IS_FACTORY_TILE] = state.board.factory_occupancy_map != -1
    obs[ObservationFeature.IS_OWN_FACTORY_TILE] = np.isin(
        state.board.factory_occupancy_map, _own_lichen_strains
    )
    obs[ObservationFeature.IS_OPPONENT_FACTORY_TILE] = np.isin(
        state.board.factory_occupancy_map, _opponent_lichen_strains
    )

    def add_unit(u: UnitStateDict, is_own: bool) -> None:
        x, y = u["pos"]
        obs[ObservationFeature.UNIT, x, y] = True
        if is_own:
            obs[ObservationFeature.OWN_UNIT, x, y] = True
        else:
            obs[ObservationFeature.OPPONENT_UNIT, x, y] = True
        _is_heavy = u["unit_type"] == "HEAVY"
        obs[ObservationFeature.UNIT_IS_HEAVY, x, y] = _is_heavy

        _cargo_space = HEAVY_ROBOT.CARGO_SPACE if _is_heavy else LIGHT_ROBOT.CARGO_SPACE

        def add_cargo(v: int, idx: int) -> None:
            obs[idx : idx + 4, x, y] = (
                v / HEAVY_ROBOT.CARGO_SPACE,
                v / _cargo_space,
                v > LIGHT_ROBOT.CARGO_SPACE,
                v == _cargo_space,
            )

        _cargo = u["cargo"]
        add_cargo(_cargo["ice"], ObservationFeature.ICE_UNIT)
        add_cargo(_cargo["ore"], ObservationFeature.ORE_UNIT)
        add_cargo(_cargo["water"], ObservationFeature.WATER_UNIT)
        add_cargo(_cargo["metal"], ObservationFeature.METAL_UNIT)

        _power = u["power"]
        _bat_cap = (
            HEAVY_ROBOT.BATTERY_CAPACITY if _is_heavy else LIGHT_ROBOT.BATTERY_CAPACITY
        )
        obs[ObservationFeature.POWER_UNIT : ObservationFeature.POWER_UNIT + 4, x, y] = (
            _power / HEAVY_ROBOT.BATTERY_CAPACITY,
            _power / _bat_cap,
            _power > LIGHT_ROBOT.BATTERY_CAPACITY,
            _power == _bat_cap,
        )
        _enqueued_action = enqueued_actions.get(u["unit_id"])
        if _enqueued_action is not None:
            a_idx = 0
            for a, a_sz in zip(_enqueued_action, UNIT_ACTION_SIZES):
                if a >= 0:
                    obs[ObservationFeature.ENQUEUED_ACTION + a_idx + a, x, y] = 1
                a_idx += a_sz

    for u in lux_obs["units"][p1].values():
        add_unit(u, True)
    for u in lux_obs["units"][p2].values():
        add_unit(u, False)

    obs[ObservationFeature.CAN_COLLIDE_WITH_FRIENDLY_UNIT] = move_validity_map > 1

    obs[ObservationFeature.GAME_PROGRESS] = (
        lux_obs["real_env_steps"] / env_cfg.max_episode_length
    )
    _day_idx = lux_obs["real_env_steps"] % env_cfg.CYCLE_LENGTH
    obs[ObservationFeature.DAY_CYCLE] = 1 - _day_idx / env_cfg.DAY_LENGTH

    obs[ObservationFeature.FACTORIES_TO_PLACE] = (
        lux_obs["teams"][p1]["factories_to_place"] / env_cfg.MAX_FACTORIES
    )

    if VERIFY:
        old_obs = np.transpose(
            _old_from_lux_observation(
                player, lux_obs, state, enqueued_actions, move_validity_map
            ),
            (2, 0, 1),
        )
        diffs = np.where(obs != old_obs)
        assert len(diffs[0]) == 0

    return obs


def _old_from_lux_observation(
    player: str,
    lux_obs: ObservationStateDict,
    state: LuxGameState,
    enqueued_actions: Dict[str, Optional[np.ndarray]],
    move_validity_map: np.ndarray,
) -> np.ndarray:
    env_cfg = state.env_cfg
    map_size = env_cfg.map_size
    LIGHT_ROBOT = env_cfg.ROBOTS["LIGHT"]
    HEAVY_ROBOT = env_cfg.ROBOTS["HEAVY"]

    p1 = player
    p2 = [p for p in state.teams if p != player][0]

    x = np.transpose(np.tile(np.linspace(-1, 1, num=map_size), (map_size, 1)))
    y = np.tile(np.linspace(-1, 1, num=map_size), (map_size, 1))

    ore = lux_obs["board"]["ore"]
    ice = lux_obs["board"]["ice"]

    _rubble = lux_obs["board"]["rubble"]
    non_zero_rubble = _rubble > 0
    rubble = _rubble / env_cfg.MAX_RUBBLE

    _lichen = lux_obs["board"]["lichen"]
    non_zero_lichen = _lichen > 0
    lichen = _lichen / env_cfg.MAX_LICHEN_PER_TILE
    spreadable_lichen = _lichen >= env_cfg.MIN_LICHEN_TO_SPREAD
    _lichen_strains = lux_obs["board"]["lichen_strains"]
    _own_lichen_strains = state.teams[p1].factory_strains
    _opponent_lichen_strains = state.teams[p2].factory_strains
    own_lichen = np.isin(_lichen_strains, _own_lichen_strains)
    opponent_lichen = np.isin(_lichen_strains, _opponent_lichen_strains)
    _lichen_counts = {
        k: v for k, v in zip(*np.unique(_lichen_strains, return_counts=True))
    }

    def zeros(dtype: Type) -> np.ndarray:
        return np.zeros((map_size, map_size), dtype=dtype)

    factory = zeros(np.bool_)
    own_factory = zeros(np.bool_)
    opponent_factory = zeros(np.bool_)
    ice_factory = zeros(np.float32)
    water_factory = zeros(np.float32)
    ore_factory = zeros(np.float32)
    metal_factory = zeros(np.float32)
    power_factory = zeros(np.float32)
    can_build_light_robot = zeros(np.bool_)  # Handled by invalid action mask?
    can_build_heavy_robot = zeros(np.bool_)  # Handled by invalid action mask?
    can_water_lichen = zeros(np.bool_)  # Handled by invalid action mask?
    day_survive_factory = zeros(np.float32)
    over_day_survive_factory = zeros(np.bool_)
    day_survive_water_factory = zeros(np.float32)
    connected_lichen_tiles = zeros(np.float32)
    connected_lichen = zeros(np.float32)

    def add_factory(f: FactoryStateDict, p_id: str, is_own: bool) -> None:
        f_state = state.factories[p_id][f["unit_id"]]
        x, y = f["pos"]
        factory[x, y] = True
        if is_own:
            own_factory[x, y] = True
        else:
            opponent_factory[x, y] = True
        _cargo = f["cargo"]
        _ice = _cargo["ice"]
        _water = _cargo["water"]
        _metal = _cargo["metal"]
        _power = f["power"]
        ice_factory[x, y] = _ice / ICE_FACTORY_MAX
        water_factory[x, y] = _water / WATER_FACTORY_MAX
        ore_factory[x, y] = _cargo["ore"] / ORE_FACTORY_MAX
        metal_factory[x, y] = _metal / METAL_FACTORY_MAX
        power_factory[x, y] = _power / POWER_FACTORY_MAX

        can_build_light_robot[x, y] = is_build_light_valid(f_state, env_cfg)
        can_build_heavy_robot[x, y] = is_build_heavy_valid(f_state, env_cfg)
        _water_lichen_cost = factory_water_cost(f_state, state, env_cfg)
        can_water_lichen[x, y] = _water > _water_lichen_cost
        _water_supply = _water + _ice / env_cfg.ICE_WATER_RATIO
        _day_water_consumption = (
            env_cfg.FACTORY_WATER_CONSUMPTION * env_cfg.CYCLE_LENGTH
        )
        day_survive_factory[x, y] = max(_water_supply / _day_water_consumption, 1)
        over_day_survive_factory[x, y] = _water_supply > _day_water_consumption
        day_survive_water_factory[x, y] = (
            _water_supply - _water_lichen_cost > _day_water_consumption
        )
        connected_lichen_tiles[x, y] = (
            _lichen_counts.get(f["strain_id"], 0) / LICHEN_TILES_FACTORY_MAX
        )
        connected_lichen[x, y] = (
            np.sum(np.where(_lichen_strains == f["strain_id"], _lichen, 0))
            / LICHEN_FACTORY_MAX
        )

    for f in lux_obs["factories"][p1].values():
        add_factory(f, p1, True)
    for f in lux_obs["factories"][p2].values():
        add_factory(f, p2, False)

    is_factory_tile = state.board.factory_occupancy_map != -1
    is_own_factory_tile = np.isin(
        state.board.factory_occupancy_map, _own_lichen_strains
    )
    is_opponent_factory_tile = np.isin(
        state.board.factory_occupancy_map, _opponent_lichen_strains
    )

    # cargo (fraction of heavy capacity), cargo capacity, exceeds light cap, full
    unit_cargo_init = lambda: np.zeros((map_size, map_size, 4), dtype=np.float32)

    unit = zeros(np.bool_)
    own_unit = zeros(np.bool_)
    opponent_unit = zeros(np.bool_)
    unit_is_heavy = zeros(np.bool_)
    ice_unit = unit_cargo_init()
    ore_unit = unit_cargo_init()
    water_unit = unit_cargo_init()
    metal_unit = unit_cargo_init()
    power_unit = unit_cargo_init()
    enqueued_action = np.full(
        (map_size, map_size, UNIT_ACTION_ENCODED_SIZE), False, dtype=np.bool_
    )

    def add_unit(u: UnitStateDict, is_own: bool) -> None:
        _u_id = u["unit_id"]
        x, y = u["pos"]
        unit[x, y] = True
        if is_own:
            own_unit[x, y] = True
        else:
            opponent_unit[x, y] = True
        _is_heavy = u["unit_type"] == "HEAVY"
        unit_is_heavy[x, y] = _is_heavy

        _cargo_space = HEAVY_ROBOT.CARGO_SPACE if _is_heavy else LIGHT_ROBOT.CARGO_SPACE

        def add_cargo(v: int) -> np.ndarray:
            return np.array(
                (
                    v / HEAVY_ROBOT.CARGO_SPACE,
                    v / _cargo_space,
                    v > LIGHT_ROBOT.CARGO_SPACE,
                    v == _cargo_space,
                )
            )

        _cargo = u["cargo"]
        ice_unit[x, y] = add_cargo(_cargo["ice"])
        ore_unit[x, y] = add_cargo(_cargo["ore"])
        water_unit[x, y] = add_cargo(_cargo["water"])
        metal_unit[x, y] = add_cargo(_cargo["metal"])
        _power = u["power"]
        _h_bat_cap = HEAVY_ROBOT.BATTERY_CAPACITY
        _l_bat_cap = LIGHT_ROBOT.BATTERY_CAPACITY
        _bat_cap = _h_bat_cap if _is_heavy else _l_bat_cap
        power_unit[x, y] = np.array(
            (
                _power / _h_bat_cap,
                _power / _bat_cap,
                _power > _l_bat_cap,
                _power == _bat_cap,
            )
        )
        _enqueued_action = enqueued_actions.get(_u_id)
        if _enqueued_action is not None:
            enqueued_action[x, y] = _unit_action_to_obs(_enqueued_action)

    can_collide_with_friendly_unit = move_validity_map > 1

    for u in lux_obs["units"][p1].values():
        add_unit(u, True)
    for u in lux_obs["units"][p2].values():
        add_unit(u, False)

    turn = (
        np.ones((map_size, map_size), dtype=np.float32)
        * lux_obs["real_env_steps"]
        / env_cfg.max_episode_length
    )
    _day_fraction = (
        lux_obs["real_env_steps"] % env_cfg.CYCLE_LENGTH / env_cfg.CYCLE_LENGTH
    )
    _day_remaining = 1 - _day_fraction * env_cfg.CYCLE_LENGTH / env_cfg.DAY_LENGTH
    day_cycle = np.ones((map_size, map_size), dtype=np.float32) * _day_remaining

    _factories_to_place = (
        lux_obs["teams"][p1]["factories_to_place"] / env_cfg.MAX_FACTORIES
    )
    factories_to_place = (
        np.ones((map_size, map_size), dtype=np.float32) * _factories_to_place
    )

    return np.concatenate(
        (
            np.expand_dims(x, axis=-1),
            np.expand_dims(y, axis=-1),
            np.expand_dims(ore, axis=-1),
            np.expand_dims(ice, axis=-1),
            np.expand_dims(non_zero_rubble, axis=-1),
            np.expand_dims(rubble, axis=-1),
            np.expand_dims(non_zero_lichen, axis=-1),
            np.expand_dims(lichen, axis=-1),
            np.expand_dims(spreadable_lichen, axis=-1),
            np.expand_dims(own_lichen, axis=-1),
            np.expand_dims(opponent_lichen, axis=-1),
            np.expand_dims(factory, axis=-1),
            np.expand_dims(own_factory, axis=-1),
            np.expand_dims(opponent_factory, axis=-1),
            np.expand_dims(ice_factory, axis=-1),
            np.expand_dims(water_factory, axis=-1),
            np.expand_dims(ore_factory, axis=-1),
            np.expand_dims(metal_factory, axis=-1),
            np.expand_dims(power_factory, axis=-1),
            np.expand_dims(can_build_light_robot, axis=-1),
            np.expand_dims(can_build_heavy_robot, axis=-1),
            np.expand_dims(can_water_lichen, axis=-1),
            np.expand_dims(day_survive_factory, axis=-1),
            np.expand_dims(over_day_survive_factory, axis=-1),
            np.expand_dims(day_survive_water_factory, axis=-1),
            np.expand_dims(connected_lichen_tiles, axis=-1),
            np.expand_dims(connected_lichen, axis=-1),
            np.expand_dims(is_factory_tile, axis=-1),
            np.expand_dims(is_own_factory_tile, axis=-1),
            np.expand_dims(is_opponent_factory_tile, axis=-1),
            np.expand_dims(unit, axis=-1),
            np.expand_dims(own_unit, axis=-1),
            np.expand_dims(opponent_unit, axis=-1),
            np.expand_dims(unit_is_heavy, axis=-1),
            ice_unit,
            ore_unit,
            water_unit,
            metal_unit,
            power_unit,
            enqueued_action,
            np.expand_dims(can_collide_with_friendly_unit, axis=-1),
            np.expand_dims(turn, axis=-1),
            np.expand_dims(day_cycle, axis=-1),
            np.expand_dims(factories_to_place, axis=-1),
        ),
        axis=-1,
        dtype=np.float32,
    )


def _unit_action_to_obs(action: np.ndarray) -> np.ndarray:
    encoded = [np.zeros(sz, dtype=np.bool_) for sz in UNIT_ACTION_SIZES]
    for e, a in zip(encoded, action):
        if a < 0:
            continue
        e[a] = True
    return np.concatenate(encoded)
