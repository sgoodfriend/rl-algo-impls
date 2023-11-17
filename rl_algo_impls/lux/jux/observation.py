from functools import partial
from typing import Dict, List, NamedTuple, Optional, Tuple

from rl_algo_impls.lux.jux.grow_zone import (
    GrowZoneCarry,
    fill_valid_regions,
    grow_own_zone,
    has_growing_zones,
)
from rl_algo_impls.lux.jux.jax_init import jax_init

jax_init()

import jax
import jax.numpy as jnp
from jux.actions import UnitActionType
from jux.config import EnvConfig, JuxBufferConfig
from jux.map.board import Board as JuxBoard
from jux.state import State as JuxState
from jux.team import Team as JuxTeam
from jux.unit import Unit as JuxUnit
from jux.unit import UnitType
from jux.unit_cargo import ResourceType
from jux.utils import imax

from rl_algo_impls.lux.actions import (
    FACTORY_ACTION_ENCODED_SIZE,
    SIMPLE_UNIT_ACTION_SIZES,
)
from rl_algo_impls.lux.jux.action_mask import get_action_mask_pick_position
from rl_algo_impls.lux.jux.agent_config import JuxAgentConfig
from rl_algo_impls.lux.jux.jit import toggleable_jit
from rl_algo_impls.lux.jux.resources import get_factory_resources, get_unit_resources
from rl_algo_impls.lux.jux.util import get_ij, get_neighbor_ij, roll_outwards
from rl_algo_impls.lux.jux.water_map import unvectorized_water_map
from rl_algo_impls.lux.obs_feature import (
    FACTORY_RESOURCE_LAMBDAS,
    ICE_WATER_FACTORY_LAMBDA,
    MIN_WATER_TO_PICKUP,
    WATER_COST_LAMBDA,
    SimpleObservationFeature,
)


@partial(toggleable_jit, static_argnums=(1, 2, 3))
def observation_and_action_mask(
    state: JuxState,
    env_cfg: EnvConfig,
    buf_cfg: JuxBufferConfig,
    agent_cfg: JuxAgentConfig,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    obs, per_position_action_mask = _observation_and_per_position_action_mask(
        state, env_cfg, buf_cfg, agent_cfg
    )
    pick_position_action_mask = get_action_mask_pick_position(state, agent_cfg)

    action_mask = {
        "per_position": per_position_action_mask.reshape(
            per_position_action_mask.shape[:-2] + (-1,)
        ).transpose(0, 2, 1),
        "pick_position": pick_position_action_mask.reshape(
            pick_position_action_mask.shape[:-2] + (-1,)
        ),
    }
    return (obs, action_mask)


@partial(toggleable_jit, static_argnums=(1, 2, 3))
def _observation_and_per_position_action_mask(
    state: JuxState,
    env_cfg: EnvConfig,
    buf_cfg: JuxBufferConfig,
    agent_cfg: JuxAgentConfig,
) -> Tuple[jax.Array, jax.Array]:
    batch_size = len(state.env_steps)
    num_envs = 2 * batch_size
    feature_shape = (num_envs, env_cfg.map_size, env_cfg.map_size)
    # Features beyond UNIT_IS_HEAVY are concatenated onto the end of the observation
    obs: List[Optional[jax.Array]] = [None] * (
        SimpleObservationFeature.UNIT_IS_HEAVY + 1
    )
    obs[SimpleObservationFeature.X] = jnp.broadcast_to(
        jnp.transpose(
            jnp.tile(jnp.linspace(-1, 1, num=env_cfg.map_size), (env_cfg.map_size, 1))
        ),
        feature_shape,
    )
    obs[SimpleObservationFeature.Y] = jnp.broadcast_to(
        jnp.tile(jnp.linspace(-1, 1, num=env_cfg.map_size), (env_cfg.map_size, 1)),
        feature_shape,
    )

    obs[SimpleObservationFeature.ICE] = jnp.repeat(state.board.ice, 2, axis=0)
    obs[SimpleObservationFeature.ORE] = jnp.repeat(state.board.ore, 2, axis=0)

    _rubble = state.board.rubble
    obs[SimpleObservationFeature.NON_ZERO_RUBBLE] = jnp.repeat(_rubble > 0, 2, axis=0)
    obs[SimpleObservationFeature.RUBBLE] = jnp.repeat(
        _rubble / env_cfg.MAX_RUBBLE, 2, axis=0
    )

    # Shape: (B, W, H) - int
    _lichen = state.board.lichen
    # Shape: (B*P, W, H) - int
    _lichen_repeated = jnp.repeat(_lichen, 2, axis=0)
    # Shape: (B*P, W, H) - float
    obs[SimpleObservationFeature.LICHEN] = (
        _lichen_repeated / env_cfg.MAX_LICHEN_PER_TILE
    )
    obs[SimpleObservationFeature.LICHEN_AT_ONE] = _lichen_repeated == 1

    MAP_RESHAPE_TUPLE = (-1, env_cfg.map_size, env_cfg.map_size)

    # Shape: (B, W, H) - bool
    _non_zero_lichen = _lichen > 0
    # Shape: (B, W, H) - int
    _lichen_strains = state.board.lichen_strains
    # Shape: (B, P, F) - int
    _factory_strains = state.teams.factory_strains
    # Shape: (B, 1, W, H, 1) - int
    _lichen_strains_bc = _lichen_strains[:, None, ..., None]
    # Shape: (B, P, 1, 1, F) - int
    _factory_strains_bc = _factory_strains[:, :, None, None, :]
    # Shape: (B, P, W, H) - bool
    _own_lichen = jnp.logical_and(
        _non_zero_lichen[:, None],
        jnp.any(_lichen_strains_bc == _factory_strains_bc, axis=-1),
    )
    # Shape: (B*P, W, H) - bool
    obs[SimpleObservationFeature.OWN_LICHEN] = jnp.reshape(
        _own_lichen,
        MAP_RESHAPE_TUPLE,
    )
    obs[SimpleObservationFeature.OPPONENT_LICHEN] = jnp.reshape(
        _own_lichen[:, ::-1],
        MAP_RESHAPE_TUPLE,
    )

    obs[SimpleObservationFeature.GAME_PROGRESS] = jnp.broadcast_to(
        jnp.repeat(state.real_env_steps / env_cfg.max_episode_length, 2, axis=0)[
            :, None, None
        ],
        feature_shape,
    )
    obs[SimpleObservationFeature.DAY_CYCLE] = jnp.broadcast_to(
        jnp.repeat(
            1
            - (jnp.maximum(state.real_env_steps, 0) % env_cfg.CYCLE_LENGTH)
            / env_cfg.DAY_LENGTH,
            2,
            axis=0,
        )[:, None, None],
        feature_shape,
    )
    obs[SimpleObservationFeature.FACTORIES_TO_PLACE] = jnp.broadcast_to(
        jnp.reshape(state.teams.factories_to_place / env_cfg.MAX_FACTORIES, (-1))[
            :, None, None
        ],
        feature_shape,
    )

    # Shape: (B, P, F, 2) - int
    _factory_pos = state.factories.pos.pos
    # Shape: (B, P, W, H) - bool
    _own_factory = (
        jnp.zeros(
            (
                _factory_pos.shape[0],
                _factory_pos.shape[1],
                env_cfg.map_size,
                env_cfg.map_size,
            ),
            dtype=jnp.bool_,
        )
        .at[
            jnp.arange(_factory_pos.shape[0])[:, None, None],
            jnp.arange(_factory_pos.shape[1])[None, :, None],
            _factory_pos[..., 0],
            _factory_pos[..., 1],
        ]
        .set(True, mode="drop")
    )
    obs[SimpleObservationFeature.OWN_FACTORY] = _own_factory.reshape(MAP_RESHAPE_TUPLE)
    obs[SimpleObservationFeature.OPPONENT_FACTORY] = _own_factory[:, ::-1].reshape(
        MAP_RESHAPE_TUPLE
    )

    # Shape: B*P, R5, W, H - int32
    _factory_resources = (
        roll_outwards(get_factory_resources(state, env_cfg), (-2, -1))
        .sum(1)
        .repeat(2, axis=0)
    )
    # Shape: B*P, R5, W, H - float
    _factory_resources_scaled = 1 - jnp.exp(
        -jnp.array(FACTORY_RESOURCE_LAMBDAS)[None, :, None, None] * _factory_resources
    )
    (
        factory_ice,
        _,
        factory_water,
        factory_metal,
        factory_power,
    ) = jax.tree_util.tree_map(
        lambda r: r.squeeze(1), jnp.split(_factory_resources, 5, axis=1)
    )

    ice_water_factory = factory_water + jnp.floor(factory_ice / env_cfg.ICE_WATER_RATIO)
    obs[SimpleObservationFeature.ICE_WATER_FACTORY] = 1 - jnp.exp(
        -ICE_WATER_FACTORY_LAMBDA * ice_water_factory
    )

    # Shape: (B, 1, W, H, 1) - int (127 [max int8] is reserved for empty tiles)
    _factory_occupancy_bc = state.board.factory_occupancy_map[:, None, :, :, None]
    # Shape: (B, W, H) - bool
    _is_factory_occupied = state.board.factory_occupancy_map != imax(
        JuxTeam.__annotations__["factory_strains"]
    )
    # Shape: (B, P, W, H)
    _own_factory_occupied = jnp.logical_and(
        _is_factory_occupied[:, None],
        jnp.any(_factory_occupancy_bc == _factory_strains_bc, axis=-1),
    )
    obs[SimpleObservationFeature.IS_OWN_FACTORY_TILE] = _own_factory_occupied.reshape(
        MAP_RESHAPE_TUPLE
    )
    obs[SimpleObservationFeature.IS_OPPONENT_FACTORY_TILE] = _own_factory_occupied[
        :, ::-1
    ].reshape(MAP_RESHAPE_TUPLE)

    # Shape: int[B, 1, W, H, 1] (max int16 is reserved for empty tiles)
    _unit_map_bc = state.board.units_map[:, None, :, :, None]
    # Shape: bool[B, W, H]
    _unit_map_occupied = state.board.units_map != imax(
        JuxBoard.__annotations__["units_map"]
    )
    # Shape: int[B, P, 1, 1, U]
    _unit_ids_bc = state.units.unit_id[:, :, None, None, :]
    # Shape: bool[B, P, W, H]
    _own_unit = jnp.logical_and(
        _unit_map_occupied[:, None],
        jnp.any(_unit_map_bc == _unit_ids_bc, axis=-1),
    )
    obs[SimpleObservationFeature.OWN_UNIT] = _own_unit.reshape(MAP_RESHAPE_TUPLE)
    obs[SimpleObservationFeature.OPPONENT_UNIT] = _own_unit[:, ::-1].reshape(
        MAP_RESHAPE_TUPLE
    )

    obs[SimpleObservationFeature.UNIT_IS_HEAVY] = jnp.repeat(
        jnp.zeros(
            (
                state.units.unit_type.shape[0],
                state.units.unit_type.shape[1],
                env_cfg.map_size,
                env_cfg.map_size,
            ),
            dtype=jnp.bool_,
        )
        .at[
            jnp.arange(state.units.unit_type.shape[0])[:, None, None],
            jnp.arange(state.units.unit_type.shape[1])[None, :, None],
            state.units.pos.pos[..., 0],
            state.units.pos.pos[..., 1],
        ]
        .set(state.units.unit_type == jnp.int8(UnitType.HEAVY), mode="drop")
        .any(1),
        2,
        axis=0,
    )

    # Shape: B, P, R5, W, H - int32
    unit_resources = get_unit_resources(state, env_cfg)

    heavy_cargo_capacity = env_cfg.ROBOTS[UnitType.HEAVY].CARGO_SPACE
    heavy_power_capacity = env_cfg.ROBOTS[UnitType.HEAVY].BATTERY_CAPACITY
    # Shape: B, P, R4, W, H - float
    cargo_of_heavy_capacity = unit_resources[:, :, :-1] / heavy_cargo_capacity
    # Shape: B, P, 1, W, H - float
    power_of_heavy_capacity = unit_resources[:, :, -1:] / heavy_power_capacity

    light_cargo_capacity = env_cfg.ROBOTS[UnitType.LIGHT].CARGO_SPACE
    light_power_capacity = env_cfg.ROBOTS[UnitType.LIGHT].BATTERY_CAPACITY
    # Shape: B, P, R4, W, H - bool
    cargo_at_light_capacity = unit_resources[:, :, :-1] >= light_cargo_capacity
    # Shape: B, P, 1, W, H - bool
    power_at_light_capacity = unit_resources[:, :, -1:] >= light_power_capacity

    # Shape: B, W, H - int32
    _cargo_capacity_at_pos = jnp.where(
        obs[SimpleObservationFeature.UNIT_IS_HEAVY][::2],
        heavy_cargo_capacity,
        light_cargo_capacity,
    )
    _power_capacity_at_pos = jnp.where(
        obs[SimpleObservationFeature.UNIT_IS_HEAVY][::2],
        heavy_power_capacity,
        light_power_capacity,
    )
    # Shape: B, P, R4, W, H - float
    cargo_of_capacity = (
        unit_resources[:, :, :-1] / _cargo_capacity_at_pos[:, None, None]
    )
    # Shape: B, P, 1, W, H - float
    power_of_capacity = (
        unit_resources[:, :, -1:] / _power_capacity_at_pos[:, None, None]
    )

    # Shape: B, P, R4, W, H - bool
    cargo_at_capacity = (
        unit_resources[:, :, :-1] == _cargo_capacity_at_pos[:, None, None]
    )
    # Shape: B, P, 1, W, H - bool
    power_at_capacity = (
        unit_resources[:, :, -1:] == _power_capacity_at_pos[:, None, None]
    )

    # Shape: B*P, R4*4, W, H - float
    unit_cargo_interleaved = (
        jnp.stack(
            (
                cargo_of_heavy_capacity,
                cargo_of_capacity,
                cargo_at_light_capacity,
                cargo_at_capacity,
            ),
            axis=3,
        )
        .reshape(
            cargo_of_heavy_capacity.shape[0],
            cargo_of_heavy_capacity.shape[1],
            cargo_of_heavy_capacity.shape[2] * 4,
            cargo_of_heavy_capacity.shape[3],
            cargo_of_heavy_capacity.shape[4],
        )
        .sum(1)
        .repeat(2, axis=0)
    )
    # Shape: B*P, 4, W, H - float
    unit_power_interleaved = (
        jnp.concatenate(
            (
                power_of_heavy_capacity,
                power_of_capacity,
                power_at_light_capacity,
                power_at_capacity,
            ),
            axis=2,
        )
        .sum(1)
        .repeat(2, axis=0)
    )

    _unit_mask = (
        state.units.team_id != imax(JuxUnit.__annotations__["team_id"])
    ).reshape(-1)
    _unit_pos = jnp.where(_unit_mask[:, None], state.units.pos.pos.reshape((-1, 2)), 0)
    _unit_batch_indices = jnp.repeat(
        jnp.arange(batch_size),
        2 * buf_cfg.MAX_N_UNITS,
    )
    enqueued_action = _enqueued_actions(
        state,
        env_cfg,
        _unit_batch_indices,
        _unit_pos,
    )

    # Action mask computation
    can_factory_build_light = jnp.logical_and(
        obs[SimpleObservationFeature.OWN_FACTORY],
        jnp.logical_and(
            factory_metal >= env_cfg.ROBOTS[UnitType.LIGHT].METAL_COST,
            factory_power >= env_cfg.ROBOTS[UnitType.LIGHT].POWER_COST,
        ),
    )
    can_factory_build_heavy = jnp.logical_and(
        obs[SimpleObservationFeature.OWN_FACTORY],
        jnp.logical_and(
            factory_metal >= env_cfg.ROBOTS[UnitType.HEAVY].METAL_COST,
            factory_power >= env_cfg.ROBOTS[UnitType.HEAVY].POWER_COST,
        ),
    )

    vectorized_water_info = jax.vmap(unvectorized_water_map, in_axes=0, out_axes=0)
    color, grow_lichen_map, connected_lichen_size = vectorized_water_info(state)
    water_cost = jnp.repeat(
        jnp.ceil(grow_lichen_map / env_cfg.LICHEN_WATERING_COST_FACTOR), 2, axis=0
    )
    can_water_lichen = jnp.logical_and(
        obs[SimpleObservationFeature.OWN_FACTORY],
        jnp.logical_and(
            factory_water > water_cost,
            ice_water_factory - water_cost
            > jnp.minimum(
                env_cfg.max_episode_length - state.real_env_steps,
                agent_cfg.min_water_to_lichen,
            ).repeat(2, axis=0)[:, None, None],
        ),
    )

    own_zone_carry = GrowZoneCarry(
        own_zone=(_own_lichen | _own_factory_occupied).reshape(MAP_RESHAPE_TUPLE),
        growing_zone=jnp.zeros(
            obs[SimpleObservationFeature.IS_OWN_FACTORY_TILE].shape, dtype=jnp.bool_
        ),
        growable_zone=(_rubble == 0).repeat(2, axis=0)
        & ~obs[SimpleObservationFeature.IS_OPPONENT_FACTORY_TILE],
    )
    own_zone_carry = grow_own_zone(own_zone_carry)
    own_zone_carry = jax.lax.while_loop(
        has_growing_zones, grow_own_zone, own_zone_carry
    )
    _, adjacent_rubble, _ = grow_own_zone(
        GrowZoneCarry(
            own_zone=own_zone_carry.own_zone,
            growing_zone=jnp.zeros_like(own_zone_carry.own_zone),
            growable_zone=(_rubble > 0).repeat(2, axis=0),
        )
    )
    # bool[B*P, W, H]
    points_of_interest = (
        obs[SimpleObservationFeature.IS_OWN_FACTORY_TILE]
        | obs[SimpleObservationFeature.OWN_LICHEN]
        | adjacent_rubble
        | obs[SimpleObservationFeature.ICE]
        | obs[SimpleObservationFeature.ORE]
        | obs[SimpleObservationFeature.OPPONENT_LICHEN]
        | obs[SimpleObservationFeature.OWN_UNIT]
    )
    # bool[B*P, W, H]
    move_zone = fill_valid_regions(points_of_interest)
    LARGE_VALUE = 2**24
    # int32[B*P, W, H]
    masked_rubble = jnp.where(
        move_zone,
        jnp.repeat(state.board.rubble, 2, axis=0).astype(jnp.int32),
        LARGE_VALUE,
    )

    # Shape: int[B*P, W, H]
    _action_queueing_cost = jnp.where(
        obs[SimpleObservationFeature.UNIT_IS_HEAVY],
        env_cfg.ROBOTS[UnitType.HEAVY].ACTION_QUEUE_POWER_COST,
        env_cfg.ROBOTS[UnitType.LIGHT].ACTION_QUEUE_POWER_COST,
    )
    # Shape: int[B*P, W, H]
    _unit_power = unit_resources[:, :, -1].sum(1).repeat(2, axis=0)
    _unit_power_available = _unit_power - _action_queueing_cost

    # Shape: bool[B*P, W, H]
    is_own_unit = obs[SimpleObservationFeature.OWN_UNIT]
    # Shape: int[B*P, W, H]
    _base_move_cost = jnp.where(
        obs[SimpleObservationFeature.UNIT_IS_HEAVY],
        env_cfg.ROBOTS[UnitType.HEAVY].MOVE_COST,
        env_cfg.ROBOTS[UnitType.LIGHT].MOVE_COST,
    )
    # Shape: int[B*P, W, H]
    _rubble_move_cost = jnp.where(
        obs[SimpleObservationFeature.UNIT_IS_HEAVY],
        env_cfg.ROBOTS[UnitType.HEAVY].RUBBLE_MOVEMENT_COST,
        env_cfg.ROBOTS[UnitType.LIGHT].RUBBLE_MOVEMENT_COST,
    )
    PAD_MAP_WIDTH = ((0, 0), (1, 1), (1, 1))
    # Shape: int[B*P, W+2, H+2]
    _rubble_padded = jnp.pad(
        masked_rubble,
        PAD_MAP_WIDTH,
        mode="constant",
        constant_values=LARGE_VALUE,
    )
    ij = get_ij(env_cfg.map_size, env_cfg.map_size)
    neighbor_ij = get_neighbor_ij(ij)

    _rubble_at_dest = _rubble_padded[:, neighbor_ij[1:, 0], neighbor_ij[1:, 1]]
    _opponent_factory_at_dest = jnp.pad(
        obs[SimpleObservationFeature.IS_OPPONENT_FACTORY_TILE],
        PAD_MAP_WIDTH,
    )[:, neighbor_ij[1:, 0], neighbor_ij[1:, 1]]
    # Shape: (B*P, D4, W, H) - bool
    unit_can_move_direction = jnp.logical_and(
        is_own_unit[:, None],
        jnp.logical_and(
            jnp.logical_not(_opponent_factory_at_dest),
            _unit_power_available[:, None]
            >= _base_move_cost[:, None]
            + jnp.floor(_rubble_at_dest * _rubble_move_cost[:, None]),
        ),
    )
    # Shape: bool[B*P, W, H]
    unit_can_move = unit_can_move_direction.any(axis=1)

    # int[B, W, H]
    init_unit_power = jnp.where(
        obs[SimpleObservationFeature.UNIT_IS_HEAVY][::2],
        env_cfg.ROBOTS[UnitType.HEAVY].INIT_POWER,
        env_cfg.ROBOTS[UnitType.LIGHT].INIT_POWER,
    )

    # Shape - (N, R5, W, H) - bool
    unit_can_transfer_resource = jnp.concatenate(
        (
            unit_resources[:, :, : ResourceType.power] > 0,
            unit_resources[:, :, ResourceType.power :] > init_unit_power[:, None, None],
        ),
        axis=2,
    ).reshape((-1,) + unit_resources.shape[2:])
    # Shape - bool[B*P, D4, W+2, H+2]
    _unit_can_move_padded = jnp.pad(
        unit_can_move_direction, ((0, 0), (0, 0), (1, 1), (1, 1))
    )
    # Shape - (D4, D4, 2, W, H) - int
    _inverse_direction_index = jnp.array((2, 3, 0, 1))
    # Shape - (N, D4, W, H) - bool
    _unit_can_move_into_from_dir = _unit_can_move_padded[
        :,
        jnp.arange(4)[..., None, None],
        neighbor_ij[_inverse_direction_index + 1, 0],
        neighbor_ij[_inverse_direction_index + 1, 1],
    ]
    # Shape - (N, D4, W+2, H+2) - bool
    _unit_can_move_into_from_dir_padded = jnp.pad(
        _unit_can_move_into_from_dir, ((0, 0), (0, 0), (1, 1), (1, 1))
    )
    # Shape - (N, D4, D4, W, H) - bool
    # dim 1 - direction unit moves to get into destination position
    # dim 2 - direction of destination
    _destination_in_dir_unit_can_into_by_dir = jnp.where(
        is_own_unit[:, None, None],
        _unit_can_move_into_from_dir_padded[
            :, :, neighbor_ij[1:, 0], neighbor_ij[1:, 1]
        ],
        False,
    )
    _other_unit_can_move_into_from_dirs_indexes = jnp.array(
        (
            (1, 2, 3),
            (2, 3, 0),
            (3, 0, 1),
            (0, 1, 2),
        )
    )
    # Shape - (N, D4, W, H) - bool
    _other_unit_can_move_into = _destination_in_dir_unit_can_into_by_dir[
        :, _other_unit_can_move_into_from_dirs_indexes, jnp.arange(4)[:, None], :, :
    ].any(2)
    # Shape - (N, D4, W, H) - bool
    _own_unit_at_destination = jnp.logical_and(
        is_own_unit[:, None],
        jnp.pad(is_own_unit, PAD_MAP_WIDTH)[:, neighbor_ij[1:, 0], neighbor_ij[1:, 1]],
    )
    _own_factory_at_destination = jnp.logical_and(
        is_own_unit[:, None],
        jnp.pad(
            obs[SimpleObservationFeature.IS_OWN_FACTORY_TILE],
            PAD_MAP_WIDTH,
        )[:, neighbor_ij[1:, 0], neighbor_ij[1:, 1]],
    )
    # Shape - (N, D4, W, H) - bool
    _own_unit_could_be_at_destination = jnp.logical_or(
        _own_unit_at_destination,
        _other_unit_can_move_into,
    )
    if not agent_cfg.disable_unit_to_unit_transfers:
        _transfer_direction_valid = (
            _own_factory_at_destination | _own_unit_could_be_at_destination
        )
    elif agent_cfg.enable_factory_to_digger_power_transfers:
        # bool[B*P, W, H]
        _own_unit_on_factory = (
            is_own_unit & obs[SimpleObservationFeature.IS_OWN_FACTORY_TILE]
        )
        # bool[N, D4, W, H]
        _can_transfer_to_digger = (
            _own_unit_on_factory[:, None]
            & _own_unit_at_destination
            & jnp.pad(enqueued_action[:, UnitActionType.DIG], PAD_MAP_WIDTH)[
                :, neighbor_ij[1:, 0], neighbor_ij[1:, 1]
            ]
        )
        # If can transfer to digger, then can't transfer to factory
        _transfer_direction_valid = jnp.where(
            _can_transfer_to_digger.any(1)[:, None],
            _can_transfer_to_digger,
            _own_factory_at_destination,
        )
        # Only allow power transfer if can transfer to digger
        unit_can_transfer_resource = jnp.where(
            _can_transfer_to_digger.any(1)[:, None],
            unit_can_transfer_resource.at[:, : ResourceType.power].set(False),
            unit_can_transfer_resource,
        )
    else:
        _transfer_direction_valid = _own_factory_at_destination
    unit_can_transfer_direction = jnp.logical_and(
        jnp.logical_and(
            is_own_unit[:, None],
            jnp.any(unit_can_transfer_resource, axis=1)[:, None],
        ),
        _transfer_direction_valid,
    )
    unit_can_transfer = jnp.any(unit_can_transfer_direction, axis=1)
    unit_can_transfer_resource = jnp.logical_and(
        unit_can_transfer_resource, unit_can_transfer[:, None]
    )

    # Shape: (B*P, R5, W, H) - bool
    _has_factory_resources = (
        _factory_resources
        > jnp.array(
            (
                0,  # ice
                0,  # ore
                MIN_WATER_TO_PICKUP,  # water
                0,  # metal
                0,  # power
            )
        )[None, :, None, None]
    )
    # Shape: (B*P, R5, W, H) - bool
    _unit_at_capacity = (
        jnp.concatenate((cargo_at_capacity, power_at_capacity), axis=2)
        .any(1)
        .repeat(2, axis=0)
    )
    # Shape: bool[B*P, R5, W, H]
    _unit_has_pickup_headroom = (
        jnp.concatenate(
            (
                ~cargo_at_capacity,
                power_of_capacity < 0.9,
            ),
            axis=2,
        )
        .any(1)
        .repeat(2, axis=0)
    )
    # bool[B*P, R5, W, H]
    unit_can_pickup_resource = jnp.logical_and(
        is_own_unit[:, None],
        jnp.logical_and(
            _has_factory_resources,
            _unit_has_pickup_headroom,
        ),
    )
    if agent_cfg.disable_cargo_pickup:
        if agent_cfg.enable_light_water_pickup:
            unit_can_pickup_resource = unit_can_pickup_resource.at[
                :, jnp.array((ResourceType.ice, ResourceType.ore, ResourceType.metal))
            ].set(False)
            unit_can_pickup_resource = unit_can_pickup_resource.at[
                :, ResourceType.water
            ].set(
                jnp.logical_and(
                    unit_can_pickup_resource[:, ResourceType.water],
                    ~obs[SimpleObservationFeature.UNIT_IS_HEAVY],
                )
            )
        else:
            unit_can_pickup_resource = unit_can_pickup_resource.at[
                :, : ResourceType.power
            ].set(False)
    unit_can_pickup = unit_can_pickup_resource.any(axis=1)

    _can_mine = jnp.logical_or(
        jnp.logical_and(
            obs[SimpleObservationFeature.ICE],
            jnp.logical_not(_unit_at_capacity[:, ResourceType.ice]),
        ),
        jnp.logical_and(
            obs[SimpleObservationFeature.ORE],
            jnp.logical_not(_unit_at_capacity[:, ResourceType.ore]),
        ),
    )
    # Shape: (B*P, W, H) - int
    _dig_power_cost = jnp.where(
        obs[SimpleObservationFeature.UNIT_IS_HEAVY],
        env_cfg.ROBOTS[UnitType.HEAVY].DIG_COST,
        env_cfg.ROBOTS[UnitType.LIGHT].DIG_COST,
    )
    # Shape: (B*P, W, H) - bool
    unit_can_dig = jnp.logical_and(
        jnp.logical_and(
            is_own_unit,
            _unit_power_available >= _dig_power_cost,
        ),
        jnp.logical_or(
            jnp.logical_or(
                adjacent_rubble,
                obs[SimpleObservationFeature.OPPONENT_LICHEN],
            ),
            _can_mine,
        ),
    )

    # Shape: (B*P, W, H) - bool
    unit_can_self_destruct = jnp.logical_and(
        jnp.logical_and(
            is_own_unit,
            jnp.logical_and(
                ~obs[SimpleObservationFeature.UNIT_IS_HEAVY],
                _unit_power_available
                >= env_cfg.ROBOTS[UnitType.LIGHT].SELF_DESTRUCT_COST,
            ),
        ),
        jnp.logical_and(
            obs[SimpleObservationFeature.OPPONENT_LICHEN],
            # Only lights can self-destruct, so only need to check if lichen can be dug
            # out by lights
            _lichen_repeated > env_cfg.ROBOTS[UnitType.LIGHT].DIG_LICHEN_REMOVED,
        ),
    )

    unit_can_recharge = is_own_unit

    # Shape: (B*P, AS, W, H) - bool
    action_mask = jnp.concatenate(
        (
            jnp.stack(
                (
                    obs[SimpleObservationFeature.OWN_FACTORY],
                    can_factory_build_light,
                    can_factory_build_heavy,
                    can_water_lichen,
                    unit_can_move,
                    unit_can_transfer,
                    unit_can_pickup,
                    unit_can_dig,
                    unit_can_self_destruct,
                    unit_can_recharge,
                ),
                axis=1,
            ),
            unit_can_move_direction,
            unit_can_transfer_direction,
            unit_can_transfer_resource,
            unit_can_pickup_resource,
        ),
        axis=1,
    )

    # Don't allow any unit action if below action queueing threshold
    action_mask = action_mask.at[:, FACTORY_ACTION_ENCODED_SIZE:].set(
        jnp.logical_and(
            action_mask[:, FACTORY_ACTION_ENCODED_SIZE:],
            (_unit_power_available >= 0)[:, None],
        )
    )
    # Don't allow factory actions if real_env_steps < 0
    action_mask = action_mask.at[:, :FACTORY_ACTION_ENCODED_SIZE].set(
        jnp.logical_and(
            action_mask[:, :FACTORY_ACTION_ENCODED_SIZE],
            (state.real_env_steps >= 0).repeat(2, axis=0)[:, None, None, None],
        )
    )

    obs[SimpleObservationFeature.WATER_COST] = 1 - jnp.exp(
        -WATER_COST_LAMBDA * water_cost
    )

    combined = jnp.stack([o for o in obs if o is not None], axis=1)
    return (
        jnp.concatenate(
            (
                combined,
                _factory_resources_scaled,
                unit_cargo_interleaved,
                unit_power_interleaved,
                enqueued_action,
                _own_unit_could_be_at_destination,
            ),
            axis=1,
        ),
        action_mask,
    )


@partial(toggleable_jit, static_argnums=(1))
def _enqueued_actions(
    state: JuxState,
    env_cfg: EnvConfig,
    batch_indices: jax.Array,
    unit_pos: jax.Array,
) -> jax.Array:
    batch_size = len(state.env_steps)
    has_actions = (state.units.action_queue.count > 0).reshape(-1)
    action_idx = state.units.action_queue.front[..., None]
    action_type = jnp.where(
        has_actions,
        jnp.take_along_axis(
            state.units.action_queue.data.action_type, action_idx, axis=-1
        ).reshape(-1),
        UnitActionType.RECHARGE,
    )
    move_direction = jnp.where(
        action_type == UnitActionType.MOVE,
        jnp.take_along_axis(
            state.units.action_queue.data.direction, action_idx, axis=-1
        ).reshape(-1),
        -1,
    )
    transfer_direction = jnp.where(
        action_type == UnitActionType.TRANSFER,
        jnp.take_along_axis(
            state.units.action_queue.data.direction, action_idx, axis=-1
        ).reshape(-1),
        -1,
    )
    transfer_resource = jnp.where(
        action_type == UnitActionType.TRANSFER,
        jnp.take_along_axis(
            state.units.action_queue.data.resource_type, action_idx, axis=-1
        ).reshape(-1),
        -1,
    )
    pickup_resource = jnp.where(
        action_type == UnitActionType.PICKUP,
        jnp.take_along_axis(
            state.units.action_queue.data.resource_type, action_idx, axis=-1
        ).reshape(-1),
        -1,
    )

    def action_zeros(action_dim: int) -> jax.Array:
        return jnp.zeros(
            (batch_size + 1, action_dim, env_cfg.map_size, env_cfg.map_size),
            dtype=jnp.bool_,
        )

    encoded_action_type = (
        action_zeros(SIMPLE_UNIT_ACTION_SIZES[0])
        .at[
            jnp.where(has_actions, batch_indices, -1),
            action_type,
            unit_pos[:, 0],
            unit_pos[:, 1],
        ]
        .set(True)[:-1]
    )
    encoded_move_dir = (
        action_zeros(SIMPLE_UNIT_ACTION_SIZES[1])
        .at[
            jnp.where(move_direction != -1, batch_indices, -1),
            move_direction,
            unit_pos[:, 0],
            unit_pos[:, 1],
        ]
        .set(True)[:-1]
    )
    encoded_transfer_dir = (
        action_zeros(SIMPLE_UNIT_ACTION_SIZES[2])
        .at[
            jnp.where(transfer_direction != -1, batch_indices, -1),
            transfer_direction,
            unit_pos[:, 0],
            unit_pos[:, 1],
        ]
        .set(True)[:-1]
    )
    encoded_transfer_resource = (
        action_zeros(SIMPLE_UNIT_ACTION_SIZES[3])
        .at[
            jnp.where(transfer_resource != -1, batch_indices, -1),
            transfer_resource,
            unit_pos[:, 0],
            unit_pos[:, 1],
        ]
        .set(True)[:-1]
    )
    encoded_pickup_resource = (
        action_zeros(SIMPLE_UNIT_ACTION_SIZES[4])
        .at[
            jnp.where(pickup_resource != -1, batch_indices, -1),
            pickup_resource,
            unit_pos[:, 0],
            unit_pos[:, 1],
        ]
        .set(True)[:-1]
    )
    return jnp.repeat(
        jnp.concatenate(
            (
                encoded_action_type,
                encoded_move_dir,
                encoded_transfer_dir,
                encoded_transfer_resource,
                encoded_pickup_resource,
            ),
            axis=1,
        ),
        2,
        axis=0,
    )
