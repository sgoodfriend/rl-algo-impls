from functools import partial

from rl_algo_impls.lux.jux.jax_init import jax_init

jax_init()

import jax
import jax.numpy as jnp
from jux.config import EnvConfig
from jux.state import State as JuxState

from rl_algo_impls.lux.jux.jit import toggleable_jit


@partial(toggleable_jit, static_argnums=(1,))
def get_factory_resources(state: JuxState, env_cfg: EnvConfig) -> jax.Array:
    # Shape: B, P, F, R4 - int32
    _stock = state.factories.cargo.stock
    # Shape: B, P, R4, W, H - int32
    factory_stock_by_pos = (
        jnp.zeros(
            (
                _stock.shape[0],
                _stock.shape[1],
                _stock.shape[-1],
                env_cfg.map_size,
                env_cfg.map_size,
            ),
            dtype=_stock.dtype,
        )
        .at[
            jnp.arange(_stock.shape[0])[:, None, None, None],
            jnp.arange(_stock.shape[1])[None, :, None, None],
            jnp.arange(_stock.shape[-1])[None, None, None, :],
            state.factories.pos.pos[..., 0][..., None],
            state.factories.pos.pos[..., 1][..., None],
        ]
        .set(_stock, mode="drop")
    )
    # Shape: (B, P, F) - int32
    _power = state.factories.power
    # Shape: (B, P, 1, W, H) - int32
    factory_power_by_pos = (
        jnp.zeros(
            (_power.shape[0], _power.shape[1], env_cfg.map_size, env_cfg.map_size),
            dtype=_power.dtype,
        )
        .at[
            jnp.arange(_power.shape[0])[:, None, None],
            jnp.arange(_power.shape[1])[None, :, None],
            state.factories.pos.pos[..., 0],
            state.factories.pos.pos[..., 1],
        ]
        .set(_power, mode="drop")
    )[:, :, None]
    return jnp.concatenate((factory_stock_by_pos, factory_power_by_pos), axis=2)


@partial(toggleable_jit, static_argnums=(1,))
def get_unit_resources(state: JuxState, env_cfg: EnvConfig) -> jax.Array:
    # Shape: B, P, U, R4 - int32
    _stock = state.units.cargo.stock
    # Shape: B, P, R4, W, H - int32
    unit_stock_by_pos = (
        jnp.zeros(
            (
                _stock.shape[0],
                _stock.shape[1],
                _stock.shape[-1],
                env_cfg.map_size,
                env_cfg.map_size,
            ),
            dtype=_stock.dtype,
        )
        .at[
            jnp.arange(_stock.shape[0])[:, None, None, None],
            jnp.arange(_stock.shape[1])[None, :, None, None],
            jnp.arange(_stock.shape[-1])[None, None, None, :],
            state.units.pos.pos[..., 0][..., None],
            state.units.pos.pos[..., 1][..., None],
        ]
        .set(_stock, mode="drop")
    )
    # Shape: (B, P, U) - int32
    _power = state.units.power
    # Shape: (B, P, 1, W, H) - int32
    unit_power_by_pos = (
        jnp.zeros(
            (_power.shape[0], _power.shape[1], env_cfg.map_size, env_cfg.map_size),
            dtype=_power.dtype,
        )
        .at[
            jnp.arange(_power.shape[0])[:, None, None],
            jnp.arange(_power.shape[1])[None, :, None],
            state.units.pos.pos[..., 0],
            state.units.pos.pos[..., 1],
        ]
        .set(_power, mode="drop")[:, :, None]
    )
    return jnp.concatenate((unit_stock_by_pos, unit_power_by_pos), axis=2)
