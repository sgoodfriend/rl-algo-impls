from functools import partial
from typing import Any, Dict, NamedTuple, Union

from rl_algo_impls.lux.jux.jax_init import jax_init

jax_init()

import jax
import jax.numpy as jnp
from jux.config import EnvConfig
from jux.state import State as JuxState

from rl_algo_impls.lux.jux.agent_config import AgentConfig
from rl_algo_impls.lux.jux.jit import toggleable_jit
from rl_algo_impls.lux.rewards import MIN_SCORE


def is_namedtuple(obs: Any) -> bool:
    return isinstance(obs, tuple) and hasattr(obs, "_asdict")


def namedtuple_to_dict(nt: NamedTuple) -> Dict[str, Union[jax.Array, dict]]:
    return {
        k: (namedtuple_to_dict(v) if is_namedtuple(v) else v)
        for k, v in nt._asdict().items()
    }


@toggleable_jit
def _get_difference_ratio(array: jax.Array, eps: Union[float, jax.Array]) -> jax.Array:
    opp_array = array[:, ::-1]
    return (array - opp_array) / (array + opp_array + eps)


@toggleable_jit
def _get_delta_difference_ratio(
    array: jax.Array, delta_array: jax.Array, eps: Union[float, jax.Array]
) -> jax.Array:
    relative = _get_difference_ratio(array, eps)
    old_array = array - delta_array
    old_relative = _get_difference_ratio(old_array, eps)
    return relative - old_relative


@toggleable_jit
def _get_relative_mapping(array: jax.Array, eps: Union[float, jax.Array]) -> jax.Array:
    difference_ratio = _get_difference_ratio(array, eps)
    return 2 * difference_ratio / (1 + jnp.abs(difference_ratio))


@toggleable_jit
def _get_delta_relative_mapping(
    array: jax.Array, delta_array: jax.Array, eps: Union[float, jax.Array]
) -> jax.Array:
    relative = _get_relative_mapping(array, eps)
    old_array = array - delta_array
    old_relative = _get_relative_mapping(old_array, eps)
    return relative - old_relative


@partial(toggleable_jit, static_argnums=(4, 5))
def get_info(
    info: Dict,
    state: JuxState,
    dones: jax.Array,
    rewards: jax.Array,
    env_cfg: EnvConfig,
    agent_cfg: AgentConfig,
) -> Dict[str, Union[jax.Array, dict]]:
    opp_score = rewards[:, ::-1]
    score_delta = rewards - opp_score
    score_reward = score_delta / (rewards + opp_score + 1 - 2 * MIN_SCORE)
    win = score_delta > 0
    loss = score_delta < 0
    win_loss = jnp.int8(win) - jnp.int8(loss)
    results_dict = {
        "WinLoss": win_loss,
        "win": win,
        "loss": loss,
        "score": rewards,
        "score_delta": score_delta,
        "score_reward": score_reward,
    }
    info["relative_stats"] = jax.tree_util.tree_map(
        _get_difference_ratio
        if agent_cfg.use_difference_ratio
        else _get_relative_mapping,
        info["stats"],
        agent_cfg.relative_stats_eps,
    )
    info["delta_relative_stats"] = jax.tree_util.tree_map(
        _get_delta_difference_ratio
        if agent_cfg.use_difference_ratio
        else _get_delta_relative_mapping,
        info["stats"],
        info["delta_stats"],
        agent_cfg.relative_stats_eps,
    )
    info = {k: namedtuple_to_dict(v) for k, v in info.items()}
    # Shape: float[B, P]
    info["stats"]["real_env_steps"] = jnp.tile(state.real_env_steps[:, None], (1, 2))
    info["delta_stats"]["real_env_steps"] = jnp.tile(
        (state.real_env_steps[:, None] > 0).astype(int), (1, 2)
    )
    info["stats"]["max_episode_length"] = jnp.tile(
        (state.real_env_steps[:, None] >= env_cfg.max_episode_length).astype(int),
        (1, 2),
    )

    info["stats"]["resources"]["opponent_lichen"] = info["stats"]["resources"][
        "lichen"
    ][:, ::-1]
    info["stats"]["resources"]["opponent_factories"] = info["stats"]["resources"][
        "factories"
    ][:, ::-1]
    info["delta_stats"]["resources"]["opponent_lichen"] = info["delta_stats"][
        "resources"
    ]["lichen"][:, ::-1]
    info["delta_stats"]["resources"]["opponent_factories"] = info["delta_stats"][
        "resources"
    ]["factories"][:, ::-1]

    info = {
        **info,
        **{f"_{k}": jnp.ones_like(dones) for k in info.keys() if not k.startswith("_")},
    }

    return {
        **info,
        "_results": dones,
        "results": jax.tree_util.tree_map(
            lambda a: jnp.where(dones, a, 0), results_dict
        ),
    }
