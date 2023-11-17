import logging
from typing import Dict

from rl_algo_impls.lux.jux.jax_init import jax_init

jax_init()

import jax
import jax.numpy as jnp
from jux.state import State as JuxState

from rl_algo_impls.lux.actions import (
    FACTORY_ACTION_ENCODED_SIZE,
    SIMPLE_UNIT_ACTION_ENCODED_SIZE,
    SIMPLE_UNIT_ACTION_SIZES,
)
from rl_algo_impls.lux.jux.util import tree_leaves_with_path
from rl_algo_impls.lux.obs_feature import SimpleObservationFeature


def assert_info_relative_stats_symmetry(info: Dict) -> None:
    for path, rel_stats_array in tree_leaves_with_path(info["relative_stats"]):
        assert ((rel_stats_array[:, 0] + rel_stats_array[:, 1]) == 0).all()
        out_of_bounds = jnp.logical_or(
            rel_stats_array > 1, rel_stats_array < -1
        ).nonzero()[0]
        if len(out_of_bounds) > 0:
            raise ValueError(
                f"relative_stats out of bounds: {path}, {out_of_bounds}, {rel_stats_array[out_of_bounds]}"
            )

    for path, delta_rel_stats_array in tree_leaves_with_path(
        info["delta_relative_stats"]
    ):
        assert ((delta_rel_stats_array[:, 0] == -delta_rel_stats_array[:, 1])).all()
        out_of_bounds = jnp.logical_or(
            delta_rel_stats_array > 2, delta_rel_stats_array < -2
        ).nonzero()[0]
        if len(out_of_bounds) > 0:
            raise ValueError(
                f"delta_relative_stats out of bounds: {path}, {out_of_bounds}, {delta_rel_stats_array[out_of_bounds]}"
            )
    assert jnp.all(
        info["stats"]["resources"]["lichen"]
        == info["stats"]["resources"]["opponent_lichen"][:, ::-1]
    )
    assert jnp.all(
        info["stats"]["resources"]["factories"]
        == info["stats"]["resources"]["opponent_factories"][:, ::-1]
    )
    assert jnp.all(
        info["delta_stats"]["resources"]["lichen"]
        == info["delta_stats"]["resources"]["opponent_lichen"][:, ::-1]
    )
    assert jnp.all(
        info["delta_stats"]["resources"]["factories"]
        == info["delta_stats"]["resources"]["opponent_factories"][:, ::-1]
    )


def assert_feature_mirroring(obs: jax.Array) -> None:
    same_features = jnp.concatenate(
        (
            jnp.arange(SimpleObservationFeature.OWN_LICHEN),
            jnp.arange(
                SimpleObservationFeature.GAME_PROGRESS,
                SimpleObservationFeature.FACTORIES_TO_PLACE,
            ),
            jnp.arange(
                SimpleObservationFeature.ICE_WATER_FACTORY,
                SimpleObservationFeature.IS_OWN_FACTORY_TILE,
            ),
            jnp.arange(
                SimpleObservationFeature.UNIT_IS_HEAVY,
                SimpleObservationFeature.OWN_UNIT_COULD_BE_IN_DIRECTION,
            ),
        )
    )
    differences_along_same_features = (
        obs[::2, same_features] != obs[1::2, same_features]
    ).nonzero()
    assert (
        len(differences_along_same_features[0]) == 0
    ), f"obs not the same for player 1 and 2: {differences_along_same_features}"

    mirrored_features = jnp.array(
        (
            (
                SimpleObservationFeature.OWN_LICHEN,
                SimpleObservationFeature.OPPONENT_LICHEN,
            ),
            (
                SimpleObservationFeature.OWN_FACTORY,
                SimpleObservationFeature.OPPONENT_FACTORY,
            ),
            (
                SimpleObservationFeature.IS_OWN_FACTORY_TILE,
                SimpleObservationFeature.IS_OPPONENT_FACTORY_TILE,
            ),
            (
                SimpleObservationFeature.OWN_UNIT,
                SimpleObservationFeature.OPPONENT_UNIT,
            ),
        )
    )
    differences_along_mirrored_features = (
        obs[::2, mirrored_features[:, 0]] != obs[1::2, mirrored_features[:, 1]]
    ).nonzero()
    assert (
        len(differences_along_mirrored_features[0]) == 0
    ), f"obs not mirrored for player 1 and 2: {differences_along_mirrored_features}"


def assert_actions_get_enqueued(
    state: JuxState,
    per_position_actions: jax.Array,  # int[B, P, W, H, A6]
    per_position_action_mask: jax.Array,  # int[B*P, W*H, AM28]
    resulting_obs: jax.Array,  # float[B*P, OBS, W, H]
) -> None:
    # bool[B, P, W, H, AM28]
    actions_mask = per_position_action_mask.reshape(
        per_position_actions.shape[:-1] + (-1,)
    )
    # bool[B, P, W, H, AM24]
    unit_actions_mask = actions_mask[..., FACTORY_ACTION_ENCODED_SIZE:]
    # bool[B, P, W, H]
    any_unit_action_allowed = unit_actions_mask[..., : SIMPLE_UNIT_ACTION_SIZES[0]].any(
        -1
    )

    # bool[B, P, W, H]
    action_allowed_by_mask = jnp.take_along_axis(
        unit_actions_mask, per_position_actions[..., 1][..., None], axis=-1
    ).squeeze(-1)
    # bool[B, P, W, H]
    mask_invalid_actions = jnp.logical_and(
        any_unit_action_allowed, ~action_allowed_by_mask
    )
    assert (
        ~mask_invalid_actions.any()
    ), f"Invalid actions: {mask_invalid_actions.nonzero()}"

    # bool[B, P, W, H, AM24]
    enqueued_actions = (
        resulting_obs[
            :,
            SimpleObservationFeature.ENQUEUED_ACTION : SimpleObservationFeature.ENQUEUED_ACTION
            + SIMPLE_UNIT_ACTION_ENCODED_SIZE,
        ]
        .astype(jnp.bool_)
        .transpose((0, 2, 3, 1))
        .reshape(per_position_actions.shape[:-1] + (-1,))
    )
    # bool[B, P, W, H]
    action_in_enqueued_actions = jnp.take_along_axis(
        enqueued_actions, per_position_actions[..., 1][..., None], axis=-1
    ).squeeze(-1)
    # bool[B, P, W, H]
    action_enqueue_failures = jnp.logical_and(
        any_unit_action_allowed, ~action_in_enqueued_actions
    )
    # if action_enqueue_failures.any():
    #     logging.warn(f"Not all actions enqueued: {action_enqueue_failures.nonzero()}")


def assert_reward_only_on_done(rewards: jax.Array, dones: jax.Array) -> None:
    assert (
        rewards[~dones] == 0
    ).all(), f"rewards not zero on dones: {rewards[~dones].nonzero()}"


def assert_winner_has_factories(obs: jax.Array, rewards: jax.Array) -> None:
    assert jnp.where(
        rewards > 0, obs[:, SimpleObservationFeature.OWN_FACTORY].any((-1, -2)), True
    ).all(), f"winner has no factories: {rewards[rewards > 0].nonzero()}"
    assert jnp.where(
        jnp.logical_and(
            rewards < 0,
            (obs[:, SimpleObservationFeature.GAME_PROGRESS] < 1).any((-1, -2)),
        ),
        ~obs[:, SimpleObservationFeature.OWN_FACTORY].any((-1, -2)),
        True,
    ).all(), f"loser has factories: {rewards[rewards < 0].nonzero()}"
