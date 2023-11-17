from typing import Any, Dict, NamedTuple, Optional

import numpy as np
import ray
from luxai_s2.env import LuxAI_S2

from rl_algo_impls.lux.agent_config import LuxAgentConfig
from rl_algo_impls.lux.rewards import LuxRewardWeights
from rl_algo_impls.lux.vec_env.lux_env_gridnet import LuxEnvGridnet
from rl_algo_impls.wrappers.vector_wrapper import (
    VecEnvResetReturn,
    VecEnvStepReturn,
    VectorWrapper,
)


class LuxRayStepReturn(NamedTuple):
    step_return: VecEnvStepReturn
    action_mask: np.ndarray


class LuxRayResetReturn(NamedTuple):
    reset_return: VecEnvResetReturn
    action_mask: np.ndarray


class LuxRayEnvProperties(NamedTuple):
    map_dim: int
    single_observation_space: Dict[str, np.ndarray]
    single_action_space: Dict[str, np.ndarray]
    action_plane_space: np.ndarray
    metadata: Dict[str, Any]
    reward_weights: LuxRewardWeights


@ray.remote
class LuxRayEnv(VectorWrapper):
    def __init__(
        self,
        bid_std_dev: float = 5,
        reward_weights: Optional[Dict[str, float]] = None,
        verify: bool = False,
        factory_ore_distance_buffer: Optional[int] = None,
        factory_ice_distance_buffer: Optional[int] = None,
        valid_spawns_mask_ore_ice_union: bool = False,
        use_simplified_spaces: bool = False,
        min_ice: int = 1,
        min_ore: int = 1,
        MAX_N_UNITS: int = 512,  # Ignore
        MAX_GLOBAL_ID: int = 2 * 512,  # Ignore
        USES_COMPACT_SPAWNS_MASK: bool = False,  # Ignore
        use_difference_ratio: bool = False,  # Ignore
        relative_stats_eps: Optional[Dict[str, Dict[str, float]]] = None,  # Ignore
        disable_unit_to_unit_transfers: bool = False,
        enable_factory_to_digger_power_transfers: bool = False,
        disable_cargo_pickup: bool = False,
        enable_light_water_pickup: bool = False,
        init_water_constant: bool = False,
        min_water_to_lichen: int = 1000,
        **kwargs,
    ) -> None:
        agent_cfg = LuxAgentConfig(
            min_ice=min_ice,
            min_ore=min_ore,
            disable_unit_to_unit_transfers=disable_unit_to_unit_transfers,
            enable_factory_to_digger_power_transfers=enable_factory_to_digger_power_transfers,
            disable_cargo_pickup=disable_cargo_pickup,
            enable_light_water_pickup=enable_light_water_pickup,
            factory_ore_distance_buffer=factory_ore_distance_buffer,
            factory_ice_distance_buffer=factory_ice_distance_buffer,
            valid_spawns_mask_ore_ice_union=valid_spawns_mask_ore_ice_union,
            init_water_constant=init_water_constant,
            min_water_to_lichen=min_water_to_lichen,
            use_simplified_spaces=use_simplified_spaces,
        )
        super().__init__(
            LuxEnvGridnet(
                LuxAI_S2(collect_stats=True, **kwargs),
                agent_cfg=agent_cfg,
                bid_std_dev=bid_std_dev,
                reward_weights=reward_weights,
                verify=verify,
                reset_on_done=False,
            )
        )

    def step(self, action: np.ndarray) -> LuxRayStepReturn:
        return LuxRayStepReturn(self.env.step(action), self.env.get_action_mask())

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> LuxRayResetReturn:
        return LuxRayResetReturn(
            self.env.reset(seed=seed, options=options), self.env.get_action_mask()
        )

    def get_properties(self) -> LuxRayEnvProperties:
        return LuxRayEnvProperties(
            map_dim=self.env.env_cfg.map_size,
            single_observation_space=self.env.single_observation_space,
            single_action_space=self.env.single_action_space,
            action_plane_space=self.env.action_plane_space,
            metadata=self.env.metadata,
            reward_weights=self.env.reward_weights,
        )

    def set_reward_weights(self, reward_weights: LuxRewardWeights) -> None:
        self.env.reward_weights = reward_weights

    def call(self, method_name: str, *args, **kwargs) -> tuple:
        result = getattr(self.env, method_name)
        if callable(result):
            result = result(*args, **kwargs)
        return (result,)
