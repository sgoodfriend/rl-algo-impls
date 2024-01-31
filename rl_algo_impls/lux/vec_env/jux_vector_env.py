from typing import Dict, Optional

from rl_algo_impls.lux.jux.jax_init import jax_init

jax_init()


import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import MultiDiscrete
from jux.config import EnvConfig, JuxBufferConfig
from jux.env import JuxEnvBatch
from jux.stats import Stats as JuxStats

from rl_algo_impls.lux.actions import SIMPLE_ACTION_SIZES
from rl_algo_impls.lux.jux.actions import step_unified
from rl_algo_impls.lux.jux.agent_config import JuxAgentConfig
from rl_algo_impls.lux.jux.reset import (
    JuxResetReturnAndValidityMask,
    masked_overwrite,
    reset_and_bid,
)
from rl_algo_impls.lux.jux.verify import (
    assert_actions_get_enqueued,
    assert_feature_mirroring,
    assert_info_relative_stats_symmetry,
    assert_reward_only_on_done,
    assert_winner_has_factories,
)
from rl_algo_impls.shared.vec_env.base_vector_env import BaseVectorEnv
from rl_algo_impls.wrappers.vector_wrapper import (
    VecEnvMaskedResetReturn,
    VecEnvResetReturn,
    VecEnvStepReturn,
)


class JuxVectorEnv(BaseVectorEnv):
    def __init__(
        self,
        num_envs: int,
        bid_std_dev: float = 5,
        reward_weights: Optional[Dict[str, float]] = None,
        verify: bool = False,
        factory_ore_distance_buffer: Optional[int] = None,
        factory_ice_distance_buffer: Optional[int] = None,
        valid_spawns_mask_ore_ice_union: bool = False,
        MAX_N_UNITS: int = 512,
        MAX_GLOBAL_ID: int = 2 * 512,
        USES_COMPACT_SPAWNS_MASK: bool = False,
        render_mode: str = "rgb_array",
        use_simplified_spaces: bool = True,
        min_ice: int = 1,
        min_ore: int = 1,
        use_difference_ratio: bool = False,
        relative_stats_eps: Optional[Dict[str, Dict[str, float]]] = None,
        disable_unit_to_unit_transfers: bool = False,
        enable_factory_to_digger_power_transfers: bool = False,
        disable_cargo_pickup: bool = False,
        enable_light_water_pickup: bool = False,
        init_water_constant: bool = False,
        min_water_to_lichen: int = 1000,
        **kwargs,
    ) -> None:
        assert num_envs % 2 == 0, f"{num_envs} must be even"
        self.num_envs = num_envs
        self.batch_size = num_envs // 2
        self.bid_std_dev = bid_std_dev
        self.reward_weights = reward_weights
        self.verify = verify
        self.render_mode = render_mode
        assert use_simplified_spaces, f"Jux only supports use_simplified_spaces True"

        env_cfg = EnvConfig(
            **{**kwargs, "USES_COMPACT_SPAWNS_MASK": USES_COMPACT_SPAWNS_MASK}
        )
        buf_cfg = JuxBufferConfig(
            MAX_N_UNITS=MAX_N_UNITS,
            MAX_GLOBAL_ID=MAX_GLOBAL_ID,
            MAX_N_FACTORIES=env_cfg.MAX_FACTORIES + 1,
            MAP_SIZE=env_cfg.map_size,
        )
        self.jux_env_batch = JuxEnvBatch(env_cfg=env_cfg, buf_cfg=buf_cfg)

        self.map_size = self.jux_env_batch.env_cfg.map_size
        max_map_distance = 2 * self.map_size
        self.agent_cfg = JuxAgentConfig(
            min_ice=min_ice,
            min_ore=min_ore,
            disable_unit_to_unit_transfers=disable_unit_to_unit_transfers,
            enable_factory_to_digger_power_transfers=enable_factory_to_digger_power_transfers,
            disable_cargo_pickup=disable_cargo_pickup,
            enable_light_water_pickup=enable_light_water_pickup,
            factory_ore_distance_buffer=factory_ore_distance_buffer
            if factory_ore_distance_buffer is not None
            else max_map_distance,
            factory_ice_distance_buffer=factory_ice_distance_buffer
            if factory_ice_distance_buffer is not None
            else max_map_distance,
            valid_spawns_mask_ore_ice_union=valid_spawns_mask_ore_ice_union,
            init_water_constant=init_water_constant,
            min_water_to_lichen=min_water_to_lichen,
            relative_stats_eps=JuxStats.epsilon(**(relative_stats_eps or {})),
            use_difference_ratio=use_difference_ratio,
        )

        self.num_map_tiles = self.map_size * self.map_size
        self.action_plane_space = MultiDiscrete(np.array(SIMPLE_ACTION_SIZES))
        single_action_space = DictSpace(
            {
                "per_position": MultiDiscrete(
                    np.array(SIMPLE_ACTION_SIZES * self.num_map_tiles)
                    .flatten()
                    .tolist()
                ),
                "pick_position": MultiDiscrete([self.num_map_tiles]),
            }
        )

        self.rng_key = jax.random.PRNGKey(np.random.randint(0, np.iinfo(np.int32).max))
        obs, _ = self.reset()
        single_obs_shape = obs.shape[1:]
        single_observation_space = Box(
            low=0, high=1, shape=single_obs_shape, dtype=np.float32
        )

        super().__init__(
            num_envs,
            single_observation_space,
            single_action_space,
        )
        self.action_mask_shape = {
            "per_position": (
                self.num_map_tiles,
                self.action_plane_space.nvec.sum(),
            ),
            "pick_position": (
                len(self.single_action_space["pick_position"].nvec),
                self.num_map_tiles,
            ),
        }

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> VecEnvResetReturn:
        self._generate_reset_buffer(seed=seed, **kwargs)
        (
            self.state,
            self._jnp_obs,
            self._jnp_action_mask,
        ), valid_board_mask = self._reset(**kwargs)
        self._masked_reset(jnp.nonzero(~valid_board_mask)[0])
        return (np.asarray(self._jnp_obs), {})

    def _generate_reset_buffer(self, *, seed: Optional[int] = None, **kwargs) -> None:
        self.reset_buffer, valid_board_mask = self._reset(seed=seed, **kwargs)
        self.reset_buffer_indexes = jnp.nonzero(valid_board_mask)[0]
        self.reset_buffer_idx = 0

    def _reset(
        self, *, seed: Optional[int] = None, **kwargs
    ) -> JuxResetReturnAndValidityMask:
        seeds = jax.random.randint(
            self.next_rng_subkey(seed),
            shape=(self.batch_size,),
            minval=0,
            maxval=np.iinfo(jnp.int32).max,
        )
        bids = jax.lax.round(
            jax.random.normal(self.next_rng_subkey(), shape=(self.batch_size, 2))
            * self.bid_std_dev
        )
        return reset_and_bid(
            self.jux_env_batch,
            self.env_cfg,
            self.buf_cfg,
            seeds,
            bids,
            self.agent_cfg,
        )

    def step(self, action: np.ndarray) -> VecEnvStepReturn:
        jax_actions = {
            "pick_position": jnp.array(
                tuple(a["pick_position"] for a in action), dtype=jnp.int16
            ).reshape(self.batch_size, 2, -1),
            "per_position": jnp.array(
                tuple(a["per_position"] for a in action), dtype=jnp.int8
            ).reshape(self.batch_size, 2, self.map_size, self.map_size, -1),
        }
        (
            new_state,
            new_jnp_obs,
            new_jnp_action_mask,
            _,
            done,
            info,
        ) = step_unified(
            self.jux_env_batch,
            self.state,
            self.env_cfg,
            self.buf_cfg,
            jax_actions,
            self._jnp_obs,
            self.agent_cfg,
        )
        rew = info["results"]["WinLoss"].reshape(-1)
        if self.verify:
            assert_info_relative_stats_symmetry(info)
            assert_feature_mirroring(new_jnp_obs)
            assert_reward_only_on_done(rew, done)
            assert_winner_has_factories(new_jnp_obs, rew)
            assert_actions_get_enqueued(
                jax_actions["per_position"],
                self._jnp_action_mask["per_position"],
            )
        self.state = new_state
        self._jnp_obs = new_jnp_obs
        self._jnp_action_mask = new_jnp_action_mask

        done_indexes = jnp.nonzero(done[::2])[0]
        if done_indexes.shape[0] > 0:
            self._masked_reset(done_indexes)

        terminated = np.asarray(done)
        info = jax.tree_util.tree_map(lambda x: np.asarray(x.reshape(-1)), info)

        return (
            np.asarray(self._jnp_obs),
            np.asarray(rew),
            terminated,
            np.zeros_like(terminated),
            info,
        )

    def masked_reset(self, env_mask: np.ndarray) -> VecEnvMaskedResetReturn:
        mapped_mask = env_mask[::2]
        assert np.all(
            mapped_mask == env_mask[1::2]
        ), "env_mask must be the same for player 1 and 2: {env_mask}"
        destination_indexes = jnp.array(np.nonzero(mapped_mask)[0])
        self._masked_reset(destination_indexes)

        destination_obs_indexes = jnp.stack(
            (destination_indexes * 2, destination_indexes * 2 + 1), axis=-1
        ).reshape(-1)
        return (
            jnp.array(self._jnp_obs[destination_obs_indexes]),
            np_action_mask(
                {
                    k: v[destination_obs_indexes]
                    for k, v in self._jnp_action_mask.items()
                }
            ),
            {},
        )

    def _masked_reset(self, destination_indexes: jax.Array):
        while destination_indexes.shape[0] > 0:
            mask_size = min(
                len(self.reset_buffer_indexes) - self.reset_buffer_idx,
                destination_indexes.shape[0],
            )
            overwrite_indexes, destination_indexes = jnp.split(
                destination_indexes, jnp.array((mask_size,), dtype=jnp.int32)
            )
            origin_indexes = self.reset_buffer_indexes[
                self.reset_buffer_idx : self.reset_buffer_idx + mask_size
            ]
            self.reset_buffer_idx += mask_size

            self.state, self._jnp_obs, self._jnp_action_mask = masked_overwrite(
                self.state,
                self._jnp_obs,
                self._jnp_action_mask,
                overwrite_indexes,
                self.reset_buffer,
                origin_indexes,
            )

            if self.reset_buffer_idx == len(self.reset_buffer_indexes):
                self._generate_reset_buffer()

    def get_action_mask(self) -> np.ndarray:
        return np_action_mask(self._jnp_action_mask)

    def call(self, method_name: str, *args, **kwargs) -> tuple:
        fn = getattr(self, method_name)
        if callable(fn):
            return tuple([fn(*args, **kwargs)] * self.num_envs)
        else:
            return tuple([fn] * self.num_envs)

    @property
    def env_cfg(self) -> EnvConfig:
        return self.jux_env_batch.env_cfg

    @property
    def buf_cfg(self) -> JuxBufferConfig:
        return self.jux_env_batch.buf_cfg

    def next_rng_subkey(self, seed: Optional[int] = None) -> jax.random.KeyArray:
        if seed is not None:
            self.rng_key, rng_subkey = jax.random.split(jax.random.PRNGKey(seed))
        else:
            self.rng_key, rng_subkey = jax.random.split(self.rng_key)
        return rng_subkey


def np_action_mask(action_mask: Dict[str, jax.Array]) -> np.ndarray:
    keys = action_mask.keys()
    return np.array(
        [{k: v for k, v in zip(keys, values)} for values in zip(*action_mask.values())]
    )
