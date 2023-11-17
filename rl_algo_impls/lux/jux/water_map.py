from typing import Tuple

from rl_algo_impls.lux.jux.jax_init import jax_init

jax_init()

import jax
import jax.numpy as jnp
import jux.map_generator.flood
import jux.tree_util
from jux.factory import Factory
from jux.map.position import Position
from jux.state import State as JuxState
from jux.unit import UnitCargo
from jux.utils import imax


def unvectorized_water_map(state: JuxState) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Run flood fill algorithm to color cells. All cells to be watered by the
    same factory will have the same color.

    Returns:
        color: int[H, W, 2].
            The first dimension represent the 'color'.The 'color' is
            represented by the coordinate of the factory a tile belongs to.
            If a tile is not connected to any factory, its color its own
            coordinate. In such a way, different lichen strains will have
            different colors.

        grow_lichen_size: int[2, F].
            The number of positions to be watered by each factory.
    """
    # The key idea here is to prepare a list of neighbors for each cell it
    # connects to when watered. neighbor_ij is a 4x2xHxW array, where the
    # first dimension is the neighbors (4 at most), the second dimension is
    # the coordinates (x,y) of neighbors.
    H, W = state.board.lichen_strains.shape

    ij = jnp.mgrid[:H, :W].astype(Position.dtype())
    delta_ij = jnp.array(
        [
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, -1],
        ],
        dtype=ij.dtype,
    )  # int[2, H, W]
    neighbor_ij = delta_ij[..., None, None] + ij[None, ...]  # int[4, 2, H, W]

    # handle map boundary.
    neighbor_ij = neighbor_ij.at[0, 0, 0, :].set(0)
    neighbor_ij = neighbor_ij.at[1, 1, :, W - 1].set(W - 1)
    neighbor_ij = neighbor_ij.at[2, 0, H - 1, :].set(H - 1)
    neighbor_ij = neighbor_ij.at[3, 1, :, 0].set(0)

    # 1. calculate strain connections.
    strains_and_factory = jnp.minimum(
        state.board.lichen_strains, state.board.factory_occupancy_map
    )  # int[H, W]

    # handle a corner case where there may be rubbles on strains when movement collision happens.
    strains_and_factory = jnp.where(
        state.board.rubble == 0, strains_and_factory, imax(strains_and_factory.dtype)
    )

    neighbor_color = strains_and_factory.at[
        (
            neighbor_ij[:, 0],
            neighbor_ij[:, 1],
        )
    ].get(mode="fill", fill_value=imax(strains_and_factory.dtype))

    connect_cond = (strains_and_factory == neighbor_color) & (
        strains_and_factory != imax(strains_and_factory.dtype)
    )  # bool[4, H, W]

    color = jux.map_generator.flood._flood_fill(  # int[H, W, 2]
        jnp.concatenate(  # int[H, W, 5, 2]
            [
                jnp.where(connect_cond[:, None], neighbor_ij, ij).transpose(
                    2, 3, 0, 1
                ),  # int[H, W, 4, 2]
                ij[None].transpose(2, 3, 0, 1),  # int[H, W, 1, 2]
            ],
            axis=-2,
        )
    )
    factory_color = color.at[state.factories.pos.x, state.factories.pos.y].get(
        mode="fill", fill_value=imax(color.dtype)
    )  # int[2, F, 2]
    connected_lichen = jnp.full(
        (H, W), fill_value=imax(Factory.id_dtype())
    )  # int[H, W]
    connected_lichen = connected_lichen.at[
        factory_color[..., 0], factory_color[..., 1]
    ].set(state.factories.unit_id, mode="drop")
    connected_lichen = connected_lichen.at[color[..., 0], color[..., 1]].get(
        mode="fill", fill_value=imax(connected_lichen.dtype)
    )

    # compute connected lichen size
    # -9 for the factory occupied cells
    connected_lichen_size = (
        jux.map_generator.flood.component_sum(UnitCargo.dtype()(1), color) - 9
    )  # int[H, W]

    # 2. handle cells to expand to.
    # 2.1 cells that are allowed to expand to, only if
    #   1. it is not a lichen strain, and
    #   2. it has no rubble, and
    #   3. it is not resource.
    allow_grow = (
        (state.board.rubble == 0)
        & ~(state.board.ice | state.board.ore)
        & (state.board.lichen_strains == imax(state.board.lichen_strains.dtype))
        & (
            state.board.factory_occupancy_map
            == imax(state.board.factory_occupancy_map.dtype)
        )
    )

    # 2.2 when a non-lichen cell connects two different strains, then it is not allowed to expand to.
    neighbor_lichen_strain = strains_and_factory[
        neighbor_ij[:, 0], neighbor_ij[:, 1]
    ]  # int[4, H, W]
    neighbor_is_lichen = neighbor_lichen_strain != imax(neighbor_lichen_strain.dtype)
    center_connects_two_different_strains = (
        strains_and_factory == imax(strains_and_factory.dtype)
    ) & (
        (
            (neighbor_lichen_strain[0] != neighbor_lichen_strain[1])
            & neighbor_is_lichen[0]
            & neighbor_is_lichen[1]
        )
        | (
            (neighbor_lichen_strain[0] != neighbor_lichen_strain[2])
            & neighbor_is_lichen[0]
            & neighbor_is_lichen[2]
        )
        | (
            (neighbor_lichen_strain[0] != neighbor_lichen_strain[3])
            & neighbor_is_lichen[0]
            & neighbor_is_lichen[3]
        )
        | (
            (neighbor_lichen_strain[1] != neighbor_lichen_strain[2])
            & neighbor_is_lichen[1]
            & neighbor_is_lichen[2]
        )
        | (
            (neighbor_lichen_strain[1] != neighbor_lichen_strain[3])
            & neighbor_is_lichen[1]
            & neighbor_is_lichen[3]
        )
        | (
            (neighbor_lichen_strain[2] != neighbor_lichen_strain[3])
            & neighbor_is_lichen[2]
            & neighbor_is_lichen[3]
        )
    )
    allow_grow = allow_grow & ~center_connects_two_different_strains

    # 2.3 calculate the strains id, if it is expanded to.
    expand_center = (connected_lichen != imax(connected_lichen.dtype)) & (
        state.board.lichen >= state.env_cfg.MIN_LICHEN_TO_SPREAD
    )
    factory_occupancy = state.factories.occupancy
    expand_center = expand_center.at[factory_occupancy.x, factory_occupancy.y].set(
        True, mode="drop"
    )
    expand_center = jnp.where(
        expand_center, connected_lichen, imax(connected_lichen.dtype)
    )
    INT_MAX = imax(expand_center.dtype)
    strain_id_if_expand = jnp.minimum(  # int[H, W]
        jnp.minimum(
            jnp.roll(expand_center, 1, axis=0).at[0, :].set(INT_MAX),
            jnp.roll(expand_center, -1, axis=0).at[-1, :].set(INT_MAX),
        ),
        jnp.minimum(
            jnp.roll(expand_center, 1, axis=1).at[:, 0].set(INT_MAX),
            jnp.roll(expand_center, -1, axis=1).at[:, -1].set(INT_MAX),
        ),
    )
    strain_id_if_expand = jnp.where(allow_grow, strain_id_if_expand, INT_MAX)

    # 3. get the final color result.
    strain_id = jnp.minimum(connected_lichen, strain_id_if_expand)  # int[H, W]
    factory_idx = state.factory_id2idx[strain_id]  # int[2, H, W]
    color = state.factories.pos.pos[
        factory_idx[..., 0], factory_idx[..., 1]
    ]  # int[H, W, 2]
    color = jnp.where(
        (strain_id == imax(strain_id.dtype))[..., None], ij.transpose(1, 2, 0), color
    )

    # 4. grow_lichen_size
    # -9 for the factory occupied cells
    grow_lichen_map = (
        jux.map_generator.flood.component_sum(UnitCargo.dtype()(1), color) - 9
    )  # int[H, W]

    return color, grow_lichen_map, connected_lichen_size
