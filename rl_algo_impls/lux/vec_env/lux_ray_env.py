from typing import Any, Dict, NamedTuple, Optional

import gym
import numpy as np
import ray
from gym import Wrapper

from rl_algo_impls.lux.rewards import LuxRewardWeights
from rl_algo_impls.lux.wrappers.lux_env_gridnet import LuxEnvGridnet
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnvStepReturn


class LuxRayStepReturn(NamedTuple):
    step_return: VecEnvStepReturn
    action_mask: np.ndarray


class LuxRayResetReturn(NamedTuple):
    obs: np.ndarray
    action_mask: np.ndarray


class LuxRayEnvProperties(NamedTuple):
    map_dim: int
    single_observation_space: Dict[str, np.ndarray]
    single_action_space: Dict[str, np.ndarray]
    action_plane_space: np.ndarray
    metadata: Dict[str, Any]
    reward_weights: LuxRewardWeights


@ray.remote
class LuxRayEnv(Wrapper):
    def __init__(
        self,
        bid_std_dev: float = 5,
        reward_weights: Optional[Dict[str, float]] = None,
        verify: bool = False,
        factory_ice_distance_buffer: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            LuxEnvGridnet(
                gym.make("LuxAI_S2-v0", collect_stats=True, **kwargs),
                bid_std_dev=bid_std_dev,
                reward_weights=reward_weights,
                verify=verify,
                factory_ice_distance_buffer=factory_ice_distance_buffer,
                seed=seed,
                reset_on_done=False,
            )
        )

    def step(self, action: np.ndarray) -> LuxRayStepReturn:
        return LuxRayStepReturn(self.env.step(action), self.env.get_action_mask())

    def reset(self) -> LuxRayResetReturn:
        return LuxRayResetReturn(self.env.reset(), self.env.get_action_mask())

    def get_properties(self) -> LuxRayEnvProperties:
        return LuxRayEnvProperties(
            map_dim=self.unwrapped.env_cfg.map_size,
            single_observation_space=self.env.single_observation_space,
            single_action_space=self.env.single_action_space,
            action_plane_space=self.env.action_plane_space,
            metadata=self.env.metadata,
            reward_weights=self.env.reward_weights,
        )

    def set_reward_weights(self, reward_weights: LuxRewardWeights) -> None:
        self.env.reward_weights = reward_weights
