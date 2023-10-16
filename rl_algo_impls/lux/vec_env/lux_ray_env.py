from typing import Any, Dict, NamedTuple, Optional

import numpy as np
import ray
from luxai_s2.env import LuxAI_S2

from rl_algo_impls.lux.rewards import LuxRewardWeights
from rl_algo_impls.lux.wrappers.lux_env_gridnet import LuxEnvGridnet
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
        factory_ice_distance_buffer: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            LuxEnvGridnet(
                LuxAI_S2(collect_stats=True, **kwargs),
                bid_std_dev=bid_std_dev,
                reward_weights=reward_weights,
                verify=verify,
                factory_ice_distance_buffer=factory_ice_distance_buffer,
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
