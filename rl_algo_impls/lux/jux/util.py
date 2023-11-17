from functools import partial
from typing import List, Tuple

from rl_algo_impls.lux.jux.jax_init import jax_init

jax_init()

import jax
import jax.numpy as jnp
from jux.map.position import direct2delta_xy

from rl_algo_impls.lux.jux.jit import toggleable_jit


@partial(toggleable_jit, static_argnums=(0, 1))
def get_ij(width: int, height: int) -> jax.Array:
    # Returns Shape (2, W, H) - int
    return jnp.mgrid[:width, :height]


@toggleable_jit
def get_neighbor_ij(ij: jax.Array) -> jax.Array:
    # Returns Shape (D5, 2, W, H) - int
    return ij[None, ...] + direct2delta_xy[:, ..., None, None] + 1


@toggleable_jit
def get_reverse_neighbor_ij(ij: jax.Array) -> jax.Array:
    # Returns Shape(D5, 2, W, H) - int
    return (
        ij[None, ...] + direct2delta_xy[jnp.array((0, 3, 4, 1, 2)), ..., None, None] + 1
    )


@partial(toggleable_jit, static_argnums=(1,))
def roll_outwards(array: jax.Array, axes: Tuple[int, ...]) -> jax.Array:
    for axis in axes:
        array = roll_outwards_1d(array, axis)
    return array


@partial(toggleable_jit, static_argnums=(1,))
def roll_outwards_1d(array: jax.Array, axis: int) -> jax.Array:
    return array + jnp.roll(array, -1, axis=axis) + jnp.roll(array, 1, axis=axis)


def tree_leaves_with_path(tree) -> List[Tuple[Tuple[str], jax.Array]]:
    leaves = []

    def _tree_leaves_with_path(tree, path):
        if isinstance(tree, dict):
            for k, v in tree.items():
                _tree_leaves_with_path(v, path + (k,))
        elif isinstance(tree, (list, tuple)):
            for i, v in enumerate(tree):
                _tree_leaves_with_path(v, path + (i,))
        else:
            leaves.append((path, tree))

    _tree_leaves_with_path(tree, ())
    return leaves
