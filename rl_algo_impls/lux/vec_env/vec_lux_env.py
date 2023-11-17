from itertools import chain
from typing import Dict, List, Optional, TypeVar

import numpy as np
from gymnasium.experimental.vector.utils import batch_space
from luxai_s2.env import LuxAI_S2

from rl_algo_impls.lux.rewards import LuxRewardWeights
from rl_algo_impls.lux.vec_env.lux_env_gridnet import LuxEnvGridnet
from rl_algo_impls.shared.vec_env.base_vector_env import BaseVectorEnv
from rl_algo_impls.wrappers.vector_wrapper import (
    VecEnvMaskedResetReturn,
    VecEnvResetReturn,
    VecEnvStepReturn,
    merge_infos,
)

VecLuxEnvSelf = TypeVar("VecLuxEnvSelf", bound="VecLuxEnv")


class VecLuxEnv(BaseVectorEnv):
    def __init__(
        self,
        num_envs: int,
        bid_std_dev: float = 5,
        reward_weights: Optional[Dict[str, float]] = None,
        verify: bool = False,
        factory_ore_distance_buffer: Optional[int] = None,  # Ignore
        factory_ice_distance_buffer: Optional[int] = None,
        valid_spawns_mask_ore_ice_union: bool = False,  # Ignore
        use_simplified_spaces: bool = False,
        min_ice: int = 1,
        min_ore: int = 1,
        MAX_N_UNITS: int = 512,  # Ignore
        MAX_GLOBAL_ID: int = 2 * 512,  # Ignore
        USES_COMPACT_SPAWNS_MASK: bool = False,  # Ignore
        use_difference_ratio: bool = False,  # Ignore
        relative_stats_eps: Optional[Dict[str, Dict[str, float]]] = None,  # Ignore
        disable_unit_to_unit_transfers: bool = False,  # Ignore
        enable_factory_to_digger_power_transfers: bool = False,  # Ignore
        disable_cargo_pickup: bool = False,  # Ignore
        enable_light_water_pickup: bool = False,  # Ignore
        init_water_constant: bool = False,  # Ignore
        min_water_to_lichen: int = 1000,  # Ignore
        **kwargs,
    ) -> None:
        assert num_envs % 2 == 0, f"{num_envs} must be even"
        self.envs = [
            LuxEnvGridnet(
                LuxAI_S2(collect_stats=True, **kwargs),
                bid_std_dev=bid_std_dev,
                reward_weights=reward_weights,
                verify=verify,
                factory_ice_distance_buffer=factory_ice_distance_buffer,
                use_simplified_spaces=use_simplified_spaces,
                min_ice=min_ice,
                min_ore=min_ore,
            )
            for _ in range(num_envs // 2)
        ]
        single_env = self.envs[0]
        map_dim = single_env.unwrapped.env_cfg.map_size
        self.num_map_tiles = map_dim * map_dim
        self.action_plane_space = single_env.action_plane_space
        self.metadata = single_env.metadata

        super().__init__(
            num_envs,
            single_env.single_observation_space,
            single_env.single_action_space,
        )

    def step(self, action: np.ndarray) -> VecEnvStepReturn:
        step_returns = [
            env.step(action[2 * idx : 2 * idx + 2]) for idx, env in enumerate(self.envs)
        ]
        obs = np.concatenate([sr[0] for sr in step_returns])
        rewards = np.concatenate([sr[1] for sr in step_returns])
        truncations = np.concatenate([sr[2] for sr in step_returns])
        terminations = np.concatenate([sr[3] for sr in step_returns])
        infos = merge_infos(self, [sr[4] for sr in step_returns], 2)
        return obs, rewards, terminations, truncations, infos

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> VecEnvResetReturn:
        if seed is not None:
            seed_rng = np.random.RandomState(seed)
            seeds = seed_rng.randint(0, np.iinfo(np.int32).max, len(self.envs))
        else:
            seeds = [None for _ in self.envs]
        reset_returns = [
            env.reset(seed=s, **kwargs) for env, s in zip(self.envs, seeds)
        ]
        return np.concatenate([rr[0] for rr in reset_returns]), merge_infos(
            self, [rr[1] for rr in reset_returns], 2
        )

    def masked_reset(self, env_mask: np.ndarray) -> VecEnvMaskedResetReturn:
        mapped_mask = env_mask[::2]
        assert np.all(
            mapped_mask == env_mask[1::2]
        ), "env_mask must be the same for player 1 and 2: {env_mask}"
        masked_envs = [env for env, m in zip(self.envs, mapped_mask) if m]
        reset_returns = [env.reset() for env in masked_envs]
        return (
            np.concatenate([rr[0] for rr in reset_returns]),
            np.concatenate([env.get_action_mask() for env in masked_envs]),
            merge_infos(self, [rr[1] for rr in reset_returns], 2),
        )

    def close_extras(self, **kwargs):
        for env in self.envs:
            env.close(**kwargs)

    def get_action_mask(self) -> np.ndarray:
        return np.concatenate([env.get_action_mask() for env in self.envs])

    @property
    def reward_weights(self) -> LuxRewardWeights:
        return self.envs[0].reward_weights

    @reward_weights.setter
    def reward_weights(self, reward_weights: LuxRewardWeights) -> None:
        for env in self.envs:
            env.reward_weights = reward_weights

    def call(self, method_name: str, *args, **kwargs) -> tuple:
        results = []
        for env in self.envs:
            fn = getattr(env, method_name)
            if callable(fn):
                results.append(fn(*args, **kwargs))
            else:
                results.append(fn)
        # Most methods assume this returns a tuple of the same length as num_envs;
        # however, this would cause double rendering. Special case for render to not
        # duplicate.
        if method_name != "render":
            results = list(chain.from_iterable([r, r] for r in results))
        return tuple(results)
