from functools import partial
from typing import Dict, NamedTuple, Union

from rl_algo_impls.lux.jux.jax_init import jax_init

jax_init()

import jax
import jax.numpy as jnp
from jux.actions import FactoryAction, JuxAction, UnitAction, UnitActionType
from jux.config import EnvConfig, JuxBufferConfig
from jux.env import JuxEnvBatch
from jux.map.position import Direction
from jux.state import State as JuxState
from jux.unified_actions import FactoryPlacementAction, UnifiedAction
from jux.unit import ResourceType, UnitType
from jux.utils import imax

from rl_algo_impls.lux.jux.agent_config import JuxAgentConfig
from rl_algo_impls.lux.jux.info import get_info
from rl_algo_impls.lux.jux.jit import toggleable_jit
from rl_algo_impls.lux.jux.observation import observation_and_action_mask
from rl_algo_impls.lux.jux.resources import get_factory_resources, get_unit_resources
from rl_algo_impls.lux.jux.util import (
    get_ij,
    get_neighbor_ij,
    get_reverse_neighbor_ij,
    roll_outwards,
)
from rl_algo_impls.lux.obs_feature import MIN_WATER_TO_PICKUP, SimpleObservationFeature


class JuxStepReturn(NamedTuple):
    state: JuxState
    obs: jax.Array
    action_mask: Dict[str, jax.Array]
    reward: jax.Array
    done: jax.Array
    info: Dict[str, Union[jax.Array, Dict]]


@partial(toggleable_jit, static_argnums=(0, 2, 3, 6))
def step_unified(
    jux_env_batch: JuxEnvBatch,
    state: JuxState,
    env_cfg: EnvConfig,
    buf_cfg: JuxBufferConfig,
    actions: Dict[str, jax.Array],
    obs: jax.Array,
    agent_cfg: JuxAgentConfig,
) -> JuxStepReturn:
    factory_placement_actions = get_factory_placement_actions(
        state, env_cfg, agent_cfg, actions
    )
    late_game_actions = get_late_game_actions(state, env_cfg, actions, obs, agent_cfg)
    state, (_, rewards, dones, info) = jux_env_batch.step_unified(
        state,
        UnifiedAction(
            factory_placement_action=factory_placement_actions,
            late_game_action=late_game_actions,
        ),
    )
    obs, action_mask = observation_and_action_mask(state, env_cfg, buf_cfg, agent_cfg)
    info = get_info(info, state, dones, rewards, env_cfg, agent_cfg)
    return JuxStepReturn(
        state=state,
        obs=obs,
        action_mask=action_mask,
        reward=rewards.reshape(-1),
        done=dones.reshape(-1),
        info=info,
    )


@partial(toggleable_jit, static_argnums=(1, 2))
def get_factory_placement_actions(
    state: JuxState,
    env_cfg: EnvConfig,
    agent_cfg: JuxAgentConfig,
    actions: Dict[str, jax.Array],
) -> FactoryPlacementAction:
    # Returns FactoryPlacementAction[B, 2]
    indexes = actions["pick_position"]
    spawn = jnp.concatenate(
        (indexes // env_cfg.map_size, indexes % env_cfg.map_size), axis=-1
    ).astype(jnp.int8)
    heavy_cost_metal = env_cfg.ROBOTS[UnitType.HEAVY].METAL_COST
    metal_left = state.teams.init_metal
    metal = (
        jnp.minimum(
            metal_left - state.teams.factories_to_place * heavy_cost_metal,
            heavy_cost_metal,
        )
        + heavy_cost_metal
    )
    return FactoryPlacementAction(
        spawn=spawn,
        water=jnp.full_like(metal, env_cfg.INIT_WATER_METAL_PER_FACTORY)
        if agent_cfg.init_water_constant
        else metal,
        metal=metal,
    )


@partial(toggleable_jit, static_argnums=(1, 4))
def get_late_game_actions(
    state: JuxState,
    env_cfg: EnvConfig,
    actions_dict: Dict[str, jax.Array],
    obs: jax.Array,
    agent_cfg: JuxAgentConfig,
) -> JuxAction:
    # Shape: (B, P, W, H, A) - int8
    actions = actions_dict["per_position"]
    batch_size = len(state.env_steps)
    # Shape: (B, P, F, W, H) - float
    obs = obs.reshape(batch_size, 2, -1, env_cfg.map_size, env_cfg.map_size)

    # Shape: (B, P, W, H) - bool
    own_units = obs[:, :, SimpleObservationFeature.OWN_UNIT].astype(bool)

    # Shape: (B, P, R5, W, H) - int
    unit_resources = get_unit_resources(state, env_cfg)
    # Shape: (B, P, W, H) - int
    _unit_power = unit_resources[:, :, ResourceType.power]
    # Shape: (B, P, U)
    _action_queueing_cost_by_unit = jnp.where(
        state.units.is_heavy(),
        env_cfg.ROBOTS[UnitType.HEAVY].ACTION_QUEUE_POWER_COST,
        env_cfg.ROBOTS[UnitType.LIGHT].ACTION_QUEUE_POWER_COST,
    )
    # Shape: (B, P, W, H)
    _action_queueing_cost_by_pos = (
        jnp.zeros(
            (
                _action_queueing_cost_by_unit.shape[0],
                _action_queueing_cost_by_unit.shape[1],
                env_cfg.map_size,
                env_cfg.map_size,
            ),
            dtype=_action_queueing_cost_by_unit.dtype,
        )
        .at[
            jnp.arange(_action_queueing_cost_by_unit.shape[0])[:, None, None],
            jnp.arange(_action_queueing_cost_by_unit.shape[1])[None, :, None],
            state.units.pos.pos[..., 0],
            state.units.pos.pos[..., 1],
        ]
        .set(_action_queueing_cost_by_unit, mode="drop")
    )
    _unit_power_available = _unit_power - _action_queueing_cost_by_pos

    actions = actions.at[..., 1].set(
        jnp.where(
            jnp.logical_and(own_units, _unit_power_available >= 0),
            actions[..., 1],
            jnp.int8(UnitActionType.DO_NOTHING),
        )
    )

    # Shape: (B, P, W, H) - bool
    stationary_unit_map = get_stationary_unit_map(obs, actions)

    # Remove self-destructing units
    stationary_unit_map = jnp.logical_and(
        stationary_unit_map,
        actions[..., 1] != UnitActionType.SELF_DESTRUCT,
    )

    # Shape: (B, P, W, H) - bool
    own_factories = obs[:, :, SimpleObservationFeature.OWN_FACTORY].astype(bool)
    # Shape: int[B, P, W, H] [-1, 2]
    factory_actions_by_pos = actions[..., 0] - 1
    factory_actions_by_pos = jnp.where(
        jnp.logical_and(
            jnp.logical_and(
                own_factories, (state.real_env_steps >= 0)[:, None, None, None]
            ),
            jnp.logical_or(
                factory_actions_by_pos == FactoryAction.WATER, ~stationary_unit_map
            ),
        ),
        factory_actions_by_pos,
        jnp.int8(FactoryAction.DO_NOTHING),
    )

    stationary_unit_map = jnp.logical_or(
        stationary_unit_map,
        jnp.logical_or(
            factory_actions_by_pos == FactoryAction.BUILD_LIGHT,
            factory_actions_by_pos == FactoryAction.BUILD_HEAVY,
        ),
    )

    # Shape (B, P, D5, W, H) - bool
    occupancy_map = get_occupancy_map(stationary_unit_map, obs, actions)
    # loop_cnt = 0
    # carry = MoveFilterCarry(stationary_unit_map, obs, actions, occupancy_map)
    # while move_filter_cond_func(carry):
    #     loop_cnt += 1
    #     if loop_cnt > 10:
    #         bs, ps, xs, ys = (carry.occupancy_map.sum(2) > 1).nonzero()
    #         raise RuntimeError(
    #             "Infinite loop in move filter: "
    #             "; ".join(
    #                 str((b, p, x, y, carry.occupancy_map[b, p, :, x, y]))
    #                 for b, p, x, y in zip(bs, ps, xs, ys)
    #             )
    #         )
    #     carry = move_filter_body_func(carry)
    stationary_unit_map, _, actions, occupancy_map = jax.lax.while_loop(
        move_filter_cond_func,
        move_filter_body_func,
        MoveFilterCarry(stationary_unit_map, obs, actions, occupancy_map),
    )

    # Shape: (5, 2, W, H) - int
    neighbor_ij = get_neighbor_ij(get_ij(*obs.shape[-2:])) - 1

    # Shape: (B, P, W, H) - bool
    own_factory_tile = obs[:, :, SimpleObservationFeature.IS_OWN_FACTORY_TILE].astype(
        bool
    )
    # Shape: (B, P, D, W, H) - bool
    own_factory_tiles_in_direction = own_factory_tile[
        :, :, neighbor_ij[1:, 0], neighbor_ij[1:, 1]
    ]
    # Shape: (B, P, W, H) - bool
    own_factory_tile_in_transfer_dir = jnp.take_along_axis(
        own_factory_tiles_in_direction, actions[..., 3][:, :, None], axis=2
    ).squeeze(2)

    if not agent_cfg.disable_unit_to_unit_transfers:
        # Shape: (B, P, W, H) - bool
        unit_at_loc = occupancy_map.sum(2) > 0
        # Shape: (B, P, 4, W, H) - bool
        unit_in_direction = unit_at_loc[:, :, neighbor_ij[1:, 0], neighbor_ij[1:, 1]]
        # Shape: bool[B, P, W, H]
        own_unit_in_transfer_dir = jnp.take_along_axis(
            unit_in_direction, actions[..., 3][:, :, None], axis=2
        ).squeeze(2)
        valid_transfer_direction = (
            own_factory_tile_in_transfer_dir | own_unit_in_transfer_dir
        )
    elif agent_cfg.enable_factory_to_digger_power_transfers:
        # bool[B, P, D4, W, H]
        stationary_unit_in_dir = stationary_unit_map[
            :, :, neighbor_ij[1:, 0], neighbor_ij[1:, 1]
        ]
        # bool[B, P, W, H]
        own_stationary_unit_in_transfer_dir = jnp.take_along_axis(
            stationary_unit_in_dir, actions[..., 3][:, :, None], axis=2
        ).squeeze(2)
        valid_transfer_direction = (
            own_factory_tile_in_transfer_dir | own_stationary_unit_in_transfer_dir
        )
    else:
        valid_transfer_direction = own_factory_tile_in_transfer_dir

    # Shape: (B, P, W, H)
    cancel_transfer = jnp.logical_and(
        actions[..., 1] == UnitActionType.TRANSFER,
        ~valid_transfer_direction,
    )
    # Cancel transfers if no target
    actions = actions.at[..., 1].set(
        jnp.where(cancel_transfer, jnp.int8(UnitActionType.RECHARGE), actions[..., 1])
    )

    # Handling Transfer
    # Shape: (B, P, R5, W, H) - int
    unit_capacity = jnp.where(
        own_units[:, :, None],
        jnp.concatenate(
            (
                jnp.broadcast_to(
                    jnp.where(
                        obs[:, :, SimpleObservationFeature.UNIT_IS_HEAVY],
                        env_cfg.ROBOTS[UnitType.HEAVY].CARGO_SPACE,
                        env_cfg.ROBOTS[UnitType.LIGHT].CARGO_SPACE,
                    )[:, :, None],
                    (batch_size, 2, 4, env_cfg.map_size, env_cfg.map_size),
                ),
                jnp.where(
                    obs[:, :, SimpleObservationFeature.UNIT_IS_HEAVY],
                    env_cfg.ROBOTS[UnitType.HEAVY].BATTERY_CAPACITY,
                    env_cfg.ROBOTS[UnitType.LIGHT].BATTERY_CAPACITY,
                )[:, :, None],
            ),
            axis=2,
        ),
        0,
    )
    # Shape: (B, P, R, W, H) - int
    unit_headroom = unit_capacity - unit_resources

    # Shape: (B, P, W, H) - int
    transfer_resource_type = jnp.where(
        actions[..., 1] == UnitActionType.TRANSFER, actions[..., 4], 0
    )
    # Shape: (B, P, 5, W, H) - int
    transferrable_amounts = unit_resources.at[:, :, ResourceType.power].add(
        jnp.where(
            obs[:, :, SimpleObservationFeature.UNIT_IS_HEAVY],
            env_cfg.ROBOTS[UnitType.HEAVY].BATTERY_CAPACITY,
            env_cfg.ROBOTS[UnitType.LIGHT].BATTERY_CAPACITY,
        )
        // -10
    )
    # Shape: (B, P, W, H) - int
    transfer_source_amount = jnp.take_along_axis(
        transferrable_amounts, transfer_resource_type[:, :, None], axis=2
    ).squeeze(2)
    if not agent_cfg.disable_unit_to_unit_transfers:
        # Shape: (B, P, R5, D5, W, H) - int
        unit_headroom_in_direction = unit_headroom[
            :, :, :, neighbor_ij[:, 0], neighbor_ij[:, 1]
        ]
        # Shape: (B, P, R5, W, H) - int
        unit_headroom_after_movement = jnp.where(
            occupancy_map[:, :, None, jnp.array((0, 3, 4, 1, 2)), :, :],
            unit_headroom_in_direction,
            0,
        ).sum(3)
        # Shape: (B, P, R5, D4, W, H) - int
        unit_headroom_in_direction_after_movement = unit_headroom_after_movement[
            :, :, :, neighbor_ij[1:, 0], neighbor_ij[1:, 1]
        ]
        # Shape: (B, P, R5, W, H) - int
        unit_headroom_in_transfer_dir = jnp.take_along_axis(
            unit_headroom_in_direction_after_movement,
            actions[..., 3][:, :, None, None, :],
            axis=3,
        ).squeeze(3)
        # Shape: (B, P, W, H) - int
        resource_headroom_in_transfer_dir = jnp.take_along_axis(
            unit_headroom_in_transfer_dir, actions[..., 4][:, :, None], axis=2
        ).squeeze(2)
        # Shape: (B, P, W, H) - int
        transfer_destination_capacity = jnp.where(
            own_factory_tile_in_transfer_dir,
            env_cfg.max_transfer_amount,
            resource_headroom_in_transfer_dir,
        )
        # Shape: (B, P, W, H) - int
        transfer_amount_by_pos = jnp.minimum(
            transfer_source_amount, transfer_destination_capacity
        )
    elif agent_cfg.enable_factory_to_digger_power_transfers:
        # int[B, P, W, H]
        power_headroom = unit_headroom[:, :, ResourceType.power]
        # int[B, P, D4, W, H]
        power_headroom_in_direction = power_headroom[
            :, :, neighbor_ij[1:, 0], neighbor_ij[1:, 1]
        ]
        # int[B, P, W, H]
        power_headroom_in_transfer_dir = jnp.where(
            actions[..., 4] == ResourceType.power,
            jnp.take_along_axis(
                power_headroom_in_direction, actions[..., 3][:, :, None], axis=2
            ).squeeze(2),
            0,
        )
        # float[B, P, W, H]
        transfer_destination_capacity = jnp.where(
            own_factory_tile_in_transfer_dir,
            env_cfg.max_transfer_amount,
            power_headroom_in_transfer_dir,
        )
        transfer_amount_by_pos = jnp.minimum(
            transfer_source_amount, transfer_destination_capacity
        )
    else:
        transfer_amount_by_pos = transfer_source_amount
    # Cancel actions with transfer_amount_by_pos <= 0
    cancel_transfer_for_no_amount = jnp.logical_and(
        actions[..., 1] == UnitActionType.TRANSFER, transfer_amount_by_pos <= 0
    )
    actions = actions.at[..., 1].set(
        jnp.where(
            cancel_transfer_for_no_amount,
            jnp.int8(UnitActionType.RECHARGE),
            actions[..., 1],
        )
    )

    # Handling Pickup
    # Shape: (B, P, 5, W, H) - int
    factory_resources = get_factory_resources(state, env_cfg)
    pickup_factory_resources = roll_outwards(factory_resources, (-2, -1))
    # Shape: (1, 1, R5, W, H) - int
    min_factory_resources = jnp.array(
        (0, 0, MIN_WATER_TO_PICKUP, 0, 0), dtype=factory_resources.dtype
    )[None, None, :, None, None]
    # int[B, P, R5, W, H]
    factory_pickupable_amount = pickup_factory_resources - min_factory_resources
    if agent_cfg.disable_cargo_pickup:
        if agent_cfg.enable_light_water_pickup:
            factory_pickupable_amount = factory_pickupable_amount.at[
                :,
                :,
                jnp.array((ResourceType.ice, ResourceType.ore, ResourceType.metal)),
            ].set(0)
            factory_pickupable_amount = factory_pickupable_amount.at[
                :, :, ResourceType.water
            ].set(
                jnp.where(
                    obs[:, :, SimpleObservationFeature.UNIT_IS_HEAVY],
                    0,
                    factory_pickupable_amount[:, :, ResourceType.water],
                )
            )
        else:
            factory_pickupable_amount = factory_pickupable_amount.at[
                :, :, : ResourceType.power
            ].set(0)
    # Shape: (B, P, R5, W, H) - int
    pickup_amount_by_resource_pos = (
        jnp.zeros_like(pickup_factory_resources)
        .at[
            jnp.arange(pickup_factory_resources.shape[0])[:, None, None, None],
            jnp.arange(pickup_factory_resources.shape[1])[None, :, None, None],
            actions[..., 5],
            jnp.arange(pickup_factory_resources.shape[-2])[None, None, :, None],
            jnp.arange(pickup_factory_resources.shape[-1])[None, None, None, :],
        ]
        .set(
            jnp.where(
                actions[..., 1] == UnitActionType.PICKUP,
                jnp.take_along_axis(
                    jnp.minimum(factory_pickupable_amount, unit_headroom),
                    actions[..., 5][:, :, None],
                    axis=2,
                ).squeeze(2),
                0,
            )
        )
    )
    # Shape: (B, P, R5, W, H) - int
    pickup_amount_neighborhood = roll_outwards(pickup_amount_by_resource_pos, (-2, -1))
    # Shape: (B, P, R5, W, H) - int
    cancel_pickup_for_overdraft = roll_outwards(
        jnp.where(
            jnp.logical_and(factory_resources, pickup_amount_neighborhood),
            factory_resources - pickup_amount_neighborhood < min_factory_resources,
            False,
        ),
        (-2, -1),
    )
    # Shape: (B, P, R5, W, H) - int
    pickup_amount_by_resource_pos = jnp.where(
        cancel_pickup_for_overdraft, 0, pickup_amount_by_resource_pos
    )
    # Shape: (B, P, W, H) - int
    pickup_amount_by_pos = pickup_amount_by_resource_pos.sum(2)

    cancel_pickup_for_no_amount = jnp.logical_and(
        actions[..., 1] == UnitActionType.PICKUP, pickup_amount_by_pos <= 0
    )
    actions = actions.at[..., 1].set(
        jnp.where(
            cancel_pickup_for_no_amount,
            jnp.int8(UnitActionType.RECHARGE),
            actions[..., 1],
        )
    )

    # Recharge clears the action queue
    # Shape: (B, P, W, H) - int
    action_type_by_pos = jnp.where(
        jnp.logical_and(own_units, actions[..., 1] != UnitActionType.RECHARGE),
        actions[..., 1],
        jnp.int8(UnitActionType.DO_NOTHING),
    )
    jux_unit_action_type = _get_attribute_by_unit(state, action_type_by_pos)

    # Recheck to make sure no units on bot built spots:
    stationary_unit_map = get_stationary_unit_map(obs, actions)
    factory_actions_by_pos = jnp.where(
        jnp.logical_or(
            factory_actions_by_pos == FactoryAction.WATER, ~stationary_unit_map
        ),
        factory_actions_by_pos,
        jnp.int8(FactoryAction.DO_NOTHING),
    )
    # int[B, P, F] [-1, 2]
    factory_actions = factory_actions_by_pos[
        jnp.arange(factory_actions_by_pos.shape[0])[:, None, None],
        jnp.arange(factory_actions_by_pos.shape[1])[None, :, None],
        state.factories.pos.pos[..., 0],
        state.factories.pos.pos[..., 1],
    ]

    unit_resource_type_by_pos = jnp.where(
        action_type_by_pos == UnitActionType.TRANSFER, actions[..., 4], 0
    ) + jnp.where(action_type_by_pos == UnitActionType.PICKUP, actions[..., 5], 0)

    action_direction_by_pos = jnp.where(
        action_type_by_pos == UnitActionType.MOVE, actions[..., 2] + 1, jnp.int8(0)
    ) + jnp.where(
        action_type_by_pos == UnitActionType.TRANSFER, actions[..., 3] + 1, jnp.int8(0)
    )
    jux_unit_direction = _get_attribute_by_unit(state, action_direction_by_pos)
    jux_unit_resource_type = _get_attribute_by_unit(state, unit_resource_type_by_pos)

    unit_amount_by_pos = (
        jnp.where(
            action_type_by_pos == UnitActionType.TRANSFER, transfer_amount_by_pos, 0
        )
        + jnp.where(
            action_type_by_pos == UnitActionType.PICKUP, pickup_amount_by_pos, 0
        )
    ).astype(jnp.int16)
    jux_unit_amount = _get_attribute_by_unit(state, unit_amount_by_pos)

    unit_action_queue_shape = jux_unit_action_type.shape + (
        env_cfg.UNIT_ACTION_QUEUE_SIZE,
    )
    unit_action_queue = UnitAction(
        action_type=jnp.full(
            unit_action_queue_shape,
            fill_value=UnitActionType.DO_NOTHING,
            dtype=jnp.int8,
        )
        .at[..., 0]
        .set(jux_unit_action_type),
        direction=jnp.zeros(unit_action_queue_shape, dtype=jnp.int8)
        .at[..., 0]
        .set(jux_unit_direction),
        resource_type=jnp.zeros(unit_action_queue_shape, dtype=jnp.int8)
        .at[..., 0]
        .set(jux_unit_resource_type),
        amount=jnp.zeros(unit_action_queue_shape, dtype=jnp.int16)
        .at[..., 0]
        .set(jux_unit_amount),
        repeat=jnp.zeros(unit_action_queue_shape, dtype=jnp.int16)
        .at[..., 0]
        .set(
            jnp.where(
                jnp.logical_or(
                    jux_unit_action_type == UnitActionType.PICKUP,
                    jux_unit_action_type == UnitActionType.TRANSFER,
                ),
                0,
                jnp.full_like(
                    jux_unit_amount, imax(UnitAction.__annotations__["repeat"])
                ),
            )
        ),
        n=jnp.zeros(unit_action_queue_shape, dtype=jnp.int16)
        .at[..., 0]
        .set(jnp.ones_like(jux_unit_amount)),
    )

    unit_action_queue_count = jnp.where(
        jux_unit_action_type == UnitActionType.DO_NOTHING, 0, 1
    ).astype(jnp.int8)

    has_enqueued_actions = (state.units.action_queue.count > 0)[..., None]
    enqueued_action_idx = state.units.action_queue.front[..., None]
    enqueued_action_type = jnp.where(
        has_enqueued_actions,
        jnp.take_along_axis(
            state.units.action_queue.data.action_type, enqueued_action_idx, axis=-1
        ),
        jnp.int8(UnitActionType.DO_NOTHING),
    ).squeeze(-1)
    enqueued_action_direction = jnp.where(
        has_enqueued_actions,
        jnp.take_along_axis(
            state.units.action_queue.data.direction, enqueued_action_idx, axis=-1
        ),
        0,
    ).squeeze(-1)
    enqueued_action_resource_type = jnp.where(
        has_enqueued_actions,
        jnp.take_along_axis(
            state.units.action_queue.data.resource_type, enqueued_action_idx, axis=-1
        ),
        0,
    ).squeeze(-1)
    enqueued_action_amount = jnp.where(
        has_enqueued_actions,
        jnp.take_along_axis(
            state.units.action_queue.data.amount, enqueued_action_idx, axis=-1
        ),
        0,
    ).squeeze(-1)

    unit_action_queue_update = jnp.logical_and(
        (jux_unit_action_type != enqueued_action_type)
        | (jux_unit_direction != enqueued_action_direction)
        | (jux_unit_resource_type != enqueued_action_resource_type)
        | (jux_unit_amount != enqueued_action_amount),
        state.units.power >= _action_queueing_cost_by_unit,
    )

    return JuxAction(
        factory_actions,
        unit_action_queue,
        unit_action_queue_count,
        unit_action_queue_update,
    )


class MoveFilterCarry(NamedTuple):
    stationary_unit_map: jax.Array  # Shape: (B, P, W, H) - bool
    obs: jax.Array  # Shape: (B, P, F, W, H) - float
    action: jax.Array  # Shape: (B, P, W, H, A) - int
    occupancy_map: jax.Array  # Shape: (B, P, D, W, H) - bool


@toggleable_jit
def move_filter_cond_func(carry: MoveFilterCarry) -> bool:
    _, _, _, occupancy_map = carry
    return jnp.any(occupancy_map.sum(2) > 1)


@toggleable_jit
def get_stationary_unit_map(obs: jax.Array, action: jax.Array) -> jax.Array:
    # Returns shape (B, P, W, H) - bool
    return jnp.logical_and(
        obs[:, :, SimpleObservationFeature.OWN_UNIT].astype(bool),
        action[..., 1] != UnitActionType.MOVE,
    )


@toggleable_jit
def get_occupancy_map(
    stationary_unit_map: jax.Array, obs: jax.Array, action: jax.Array
) -> jax.Array:
    # Returns shape (B, P, D, W, H) - bool
    # Shape: (B, P, W, H) - D5
    direction_idx = jnp.where(
        action[..., 1] == UnitActionType.MOVE,
        action[..., 2] + 1,
        0,
    )

    # Shape: (D5, 2, W, H) - int
    neighbor_ij = get_neighbor_ij(get_ij(*obs.shape[-2:])) - 1

    # Shape: (B, P, 2, W, H) - int
    neighbor_idx = jnp.take_along_axis(
        neighbor_ij[None, None, :], direction_idx[:, :, None, None, :], axis=2
    ).squeeze(2)
    # Shape: (B, P, D5, W, H) - bool
    # This is a move_into_map, stationary_unit_map needs to overwrite the center direction.
    occupancy_map = (
        jnp.zeros((*obs.shape[:2], 5, *obs.shape[-2:]), dtype=bool)
        .at[
            jnp.arange(obs.shape[0])[:, None, None, None],
            jnp.arange(obs.shape[1])[None, :, None, None],
            direction_idx,
            neighbor_idx[:, :, 0],
            neighbor_idx[:, :, 1],
        ]
        .set(True)
    )
    occupancy_map = occupancy_map.at[:, :, 0].set(stationary_unit_map)
    return occupancy_map


@toggleable_jit
def move_filter_body_func(carry: MoveFilterCarry) -> MoveFilterCarry:
    stationary_unit_map, obs, action, occupancy_map = carry
    # Shape: (B, P, W, H) - bool
    cancel_destinations = occupancy_map.sum(2) > 1

    # Shape: (D, 2, W, H) - int
    reverse_neighbor_ij = get_reverse_neighbor_ij(get_ij(*obs.shape[-2:])) - 1
    # Shape: (B, P, D5, W, H) - bool
    cancel_action_at = (
        jnp.zeros(
            (*obs.shape[:2], reverse_neighbor_ij.shape[0], *obs.shape[-2:]), dtype=bool
        )
        .at[
            jnp.arange(obs.shape[0])[:, None, None, None, None],
            jnp.arange(obs.shape[1])[None, :, None, None, None],
            jnp.arange(reverse_neighbor_ij.shape[0])[None, None, :, None, None],
            reverse_neighbor_ij[:, 0][None, None, :],
            reverse_neighbor_ij[:, 1][None, None, :],
        ]
        .set(cancel_destinations[:, :, None])
    )
    # Shape: (B, P, W, H) - D5
    move_direction = (
        jnp.where(action[..., 1] == UnitActionType.MOVE, action[..., 2], -1) + 1
    )
    # Shape: (B, P, W, H) - bool
    cancel_move = jnp.where(
        move_direction != Direction.CENTER,
        jnp.take_along_axis(cancel_action_at, move_direction[:, :, None], 2).squeeze(2),
        False,
    )
    action = action.at[..., 1].set(
        jnp.where(cancel_move, jnp.int8(UnitActionType.RECHARGE), action[..., 1])
    )
    stationary_unit_map = jnp.logical_or(
        stationary_unit_map,
        jnp.logical_and(
            obs[:, :, SimpleObservationFeature.OWN_UNIT].astype(bool),
            action[..., 1] != UnitActionType.MOVE,
        ),
    )
    occupancy_map = get_occupancy_map(stationary_unit_map, obs, action)
    return MoveFilterCarry(stationary_unit_map, obs, action, occupancy_map)


@toggleable_jit
def _get_attribute_by_unit(state: JuxState, attribute: jax.Array) -> jax.Array:
    return attribute[
        jnp.arange(attribute.shape[0])[:, None, None],
        jnp.arange(attribute.shape[1])[None, :, None],
        state.units.pos.pos[..., 0],
        state.units.pos.pos[..., 1],
    ]
