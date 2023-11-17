from typing import NamedTuple

from rl_algo_impls.lux.jux.jax_init import jax_init

jax_init()

import jax
import jax.numpy as jnp

from rl_algo_impls.lux.jux.jit import toggleable_jit


class GrowZoneCarry(NamedTuple):
    own_zone: jax.Array  # bool[B*P, W, H]
    growing_zone: jax.Array  # bool[B*P, W, H]
    growable_zone: jax.Array  # bool[B*P, W, H]


@toggleable_jit
def has_growing_zones(carry: GrowZoneCarry) -> jax.Array:  # bool[1]
    return carry.growing_zone.any()


@toggleable_jit
def grow_own_zone(carry: GrowZoneCarry) -> GrowZoneCarry:
    own_zone, growing_zone, growable_zone = carry
    # bool[B*P, W, H]
    spread_zone = jnp.stack(
        (
            own_zone.at[:, :, :-1].set(own_zone[:, :, 1:]),  # up
            own_zone.at[:, 1:, :].set(own_zone[:, :-1, :]),  # right
            own_zone.at[:, :, 1:].set(own_zone[:, :, :-1]),  # down
            own_zone.at[:, :-1, :].set(own_zone[:, 1:, :]),  # left
        ),
        axis=-1,
    ).any(-1)
    growing_zone = spread_zone & ~own_zone & growable_zone
    own_zone = own_zone | growing_zone
    return GrowZoneCarry(own_zone, growing_zone, growable_zone)


def fill_valid_regions(valid_masks: jax.Array) -> jax.Array:
    # int[B*P, 1, H]
    x_left = jnp.where(
        valid_masks.any(-2, keepdims=True),
        jnp.argmax(valid_masks, axis=-2, keepdims=True),
        valid_masks.shape[-1],
    )
    # int[B*P, W, 1]
    y_up = jnp.where(
        valid_masks.any(-1, keepdims=True),
        jnp.argmax(valid_masks, axis=-1, keepdims=True),
        valid_masks.shape[-2],
    )
    x_right = jnp.where(
        valid_masks.any(-2, keepdims=True),
        valid_masks.shape[-1]
        - jnp.argmax(valid_masks[..., ::-1, :], axis=-2, keepdims=True),
        0,
    )
    y_down = jnp.where(
        valid_masks.any(-1, keepdims=True),
        valid_masks.shape[-2]
        - jnp.argmax(valid_masks[..., ::-1], axis=-1, keepdims=True),
        0,
    )
    # bool[B*P, W, 1]
    x_fill = (
        jnp.arange(valid_masks.shape[-2])[None, :, None]
        >= x_left.min(-1, keepdims=True)
    ) & (
        jnp.arange(valid_masks.shape[-2])[None, :, None]
        < x_right.max(-1, keepdims=True)
    )
    # bool[B*P, 1, H]
    y_fill = (
        jnp.arange(valid_masks.shape[-1])[None, None, :] >= y_up.min(-2, keepdims=True)
    ) & (
        jnp.arange(valid_masks.shape[-1])[None, None, :] < y_down.max(-2, keepdims=True)
    )
    # bool[B*P, W, H]
    filled_valid_masks = x_fill & y_fill
    return filled_valid_masks
