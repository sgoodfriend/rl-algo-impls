import logging
from functools import partial
from typing import NamedTuple

from rl_algo_impls.lux.jux.jax_init import jax_init

jax_init()

import jax
import jax.numpy as jnp
from jux.state import State as JuxState

from rl_algo_impls.lux.jux.agent_config import JuxAgentConfig
from rl_algo_impls.lux.jux.jit import toggleable_jit
from rl_algo_impls.lux.jux.util import roll_outwards


@partial(toggleable_jit, static_argnums=(1,))
def get_action_mask_pick_position(
    state: JuxState, agent_cfg: JuxAgentConfig
) -> jax.Array:
    valid_spawns_mask = state.board.valid_spawns_mask
    max_map_dist = valid_spawns_mask.shape[-1] + valid_spawns_mask.shape[-2] - 2
    if agent_cfg.valid_spawns_mask_ore_ice_union:
        valid_spawns_mask = jnp.logical_or(
            get_valid_resource_mask(
                valid_spawns_mask,
                state.teams.factories_to_place,
                state.board.ore,
                agent_cfg.factory_ore_distance_buffer,
            ),
            get_valid_resource_mask(
                valid_spawns_mask,
                state.teams.factories_to_place,
                state.board.ice,
                agent_cfg.factory_ice_distance_buffer,
            ),
        )
    else:
        if agent_cfg.factory_ore_distance_buffer < max_map_dist:
            valid_spawns_mask = get_valid_resource_mask(
                valid_spawns_mask,
                state.teams.factories_to_place,
                state.board.ore,
                agent_cfg.factory_ore_distance_buffer,
            )
        if agent_cfg.factory_ice_distance_buffer < max_map_dist:
            valid_spawns_mask = get_valid_resource_mask(
                valid_spawns_mask,
                state.teams.factories_to_place,
                state.board.ice,
                agent_cfg.factory_ice_distance_buffer,
            )

    return jnp.where(
        jnp.logical_and(
            state.teams.factories_to_place > 0,
            state.next_player[:, None] == jnp.arange(2),
        )[..., None, None],
        valid_spawns_mask[:, None],
        False,
    ).reshape(
        (
            -1,
            valid_spawns_mask.shape[-2],
            valid_spawns_mask.shape[-1],
        )
    )[
        :, None
    ]


@partial(toggleable_jit)
def get_valid_resource_mask(
    valid_spawns_mask: jax.Array,
    factories_to_place: jax.Array,
    resource_map: jax.Array,
    min_distance_buffer: int,
):
    expanded_resource_map = roll_outwards(
        jnp.pad(resource_map, ((0, 0), (1, 1), (1, 1))), (-2, -1)
    )[:, 1:-1, 1:-1]
    # Shape: float[B, W, H]
    distance_map = jnp.where(
        jnp.logical_and((factories_to_place > 0).any(1), resource_map.any((1, 2)))[
            :, None, None
        ],
        jnp.where(
            expanded_resource_map,
            0,
            jnp.full(valid_spawns_mask.shape, jnp.inf),
        ),
        -min_distance_buffer,  # Trick to make has_incomplete_dist_maps return out early
    )

    _, masked_dist_map, _, _, _ = jax.lax.while_loop(
        has_incomplete_dist_maps,
        step_masked_dist_map,
        MaskedDistMapCarry(
            distance_map,
            jnp.where(valid_spawns_mask, distance_map, jnp.inf),
            0,
            valid_spawns_mask,
            min_distance_buffer,
        ),
    )
    max_distance = jnp.min(masked_dist_map, axis=(1, 2)) + min_distance_buffer
    valid_spawns_mask = masked_dist_map <= max_distance[:, None, None]

    # Shape: bool[B, W, H]
    return valid_spawns_mask


class MaskedDistMapCarry(NamedTuple):
    dist_map: jax.Array
    masked_dist_map: jax.Array
    steps: int
    valid_spawns_mask: jax.Array
    distance_buffer: int


@toggleable_jit
def has_incomplete_dist_maps(carry: MaskedDistMapCarry) -> jax.Array:
    _, masked_dist_map, steps, _, distance_buffer = carry
    return jnp.any(
        steps
        < jnp.minimum(
            jnp.min(masked_dist_map, axis=(1, 2)) + distance_buffer,
            masked_dist_map.shape[-1] + masked_dist_map.shape[-2] - 2,
        )
    )


@toggleable_jit
def step_masked_dist_map(carry: MaskedDistMapCarry) -> MaskedDistMapCarry:
    dist_map, masked_dist_map, steps, valid_spawns_mask, distance_buffer = carry
    dist_map = jnp.min(
        jnp.stack(
            (
                dist_map,
                dist_map.at[:, :, :-1].set(1 + dist_map[:, :, 1:]),  # up
                dist_map.at[:, 1:, :].set(1 + dist_map[:, :-1, :]),  # right
                dist_map.at[:, :, 1:].set(1 + dist_map[:, :, :-1]),  # down
                dist_map.at[:, :-1, :].set(1 + dist_map[:, 1:, :]),  # left
            ),
            axis=-1,
        ),
        axis=-1,
    )
    masked_dist_map = jnp.where(valid_spawns_mask, dist_map, jnp.inf)
    steps += 1
    return MaskedDistMapCarry(
        dist_map, masked_dist_map, steps, valid_spawns_mask, distance_buffer
    )
