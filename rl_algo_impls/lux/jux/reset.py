from functools import partial
from typing import Dict, NamedTuple

from rl_algo_impls.lux.jux.jax_init import jax_init

jax_init()

import jax
import jax.numpy as jnp
from jux.config import EnvConfig, JuxBufferConfig
from jux.env import JuxEnvBatch
from jux.state import State as JuxState
from jux.stats import Stats as JuxStats

from rl_algo_impls.lux.jux.actions import JuxStepReturn
from rl_algo_impls.lux.jux.agent_config import AgentConfig
from rl_algo_impls.lux.jux.info import get_info
from rl_algo_impls.lux.jux.jit import toggleable_jit
from rl_algo_impls.lux.jux.observation import observation_and_action_mask


class JuxResetReturn(NamedTuple):
    state: JuxState
    obs: jax.Array
    action_mask: Dict[str, jax.Array]


@partial(toggleable_jit, static_argnums=(0, 2, 3, 5))
def _step_bid(
    jux_env_batch: JuxEnvBatch,
    state: JuxState,
    env_cfg: EnvConfig,
    buf_cfg: JuxBufferConfig,
    bids: jax.Array,
    agent_cfg: AgentConfig,
) -> JuxStepReturn:
    batch_size = len(state.env_steps)
    factions = jnp.tile(jnp.arange(2), (batch_size, 2))
    state, (_, rewards, dones, info) = jux_env_batch.step_bid(state, bids, factions)
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


class JuxResetReturnAndValidityMask(NamedTuple):
    reset_return: JuxResetReturn
    valid_mask: jax.Array


@partial(toggleable_jit, static_argnums=(0, 1, 2, 5))
def reset_and_bid(
    jux_env_batch: JuxEnvBatch,
    env_cfg: EnvConfig,
    buf_cfg: JuxBufferConfig,
    seeds: jax.Array,
    bids: jax.Array,
    agent_cfg: AgentConfig,
) -> JuxResetReturnAndValidityMask:
    state = jux_env_batch.reset(seeds, same_factories_per_team=False)
    state, obs, action_mask, _, _, _ = _step_bid(
        jux_env_batch, state, env_cfg, buf_cfg, bids, agent_cfg
    )
    return JuxResetReturnAndValidityMask(
        JuxResetReturn(state, obs, action_mask),
        jnp.logical_and(
            state.board.ice.sum((-2, -1)) >= agent_cfg.min_ice,
            state.board.ore.sum((-2, -1)) >= agent_cfg.min_ore,
        ),
    )


@toggleable_jit
def masked_overwrite(
    destination_state: JuxState,
    destination_obs: jax.Array,
    destination_action_mask: Dict[str, jax.Array],
    destination_indexes: jax.Array,
    origin_reset_return: JuxResetReturn,
    origin_indexes: jax.Array,
) -> JuxResetReturn:
    assert jnp.issubdtype(
        destination_indexes.dtype, jnp.integer
    ), f"destination_indexes must be integer dtype. destination_indexes.dtype={destination_indexes.dtype}"
    assert jnp.issubdtype(
        origin_indexes.dtype, jnp.integer
    ), f"origin_indexes must be integer dtype. origin_indexes.dtype={origin_indexes.dtype}"
    assert destination_indexes.shape == origin_indexes.shape, (
        "destination_indexes and origin_indexes must be same shape. "
        f"destination_indexes.shape={destination_indexes.shape}, "
        f"origin_indexes.shape={origin_indexes.shape}"
    )

    @toggleable_jit
    def masked_replace(
        destination_array: jax.Array,
        destination_indexes: jax.Array,
        origin_array: jax.Array,
        origin_indexes: jax.Array,
    ):
        return destination_array.at[destination_indexes].set(
            origin_array[origin_indexes]
        )

    destination_state = jax.tree_util.tree_map(
        lambda dest, origin: masked_replace(
            dest, destination_indexes, origin, origin_indexes
        ),
        destination_state,
        origin_reset_return.state,
    )

    destination_obs_indexes = jnp.stack(
        (destination_indexes * 2, destination_indexes * 2 + 1), axis=-1
    ).reshape(-1)
    origin_obs_indexes = jnp.stack(
        (origin_indexes * 2, origin_indexes * 2 + 1), axis=-1
    ).reshape(-1)
    destination_obs = destination_obs.at[destination_obs_indexes].set(
        origin_reset_return.obs[origin_obs_indexes]
    )
    destination_action_mask = jax.tree_util.tree_map(
        lambda dest, origin: masked_replace(
            dest, destination_obs_indexes, origin, origin_obs_indexes
        ),
        destination_action_mask,
        origin_reset_return.action_mask,
    )
    return JuxResetReturn(
        state=destination_state,
        obs=destination_obs,
        action_mask=destination_action_mask,
    )
