import gym
import numpy as np
from gym.spaces import Box, MultiDiscrete
from gym.spaces import Tuple as TupleSpace
from gym.vector.utils import batch_space
from gym.vector.vector_env import VectorEnv

from rl_algo_impls.shared.lux.actions import ACTION_SIZES
from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnvObs,
    VecEnvStepReturn,
    single_action_space,
    single_observation_space,
)


class VecLuxEnvGridnetWrapper(gym.Wrapper):
    def __init__(self, env: VectorEnv) -> None:
        super().__init__(env)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        assert self.is_vector_env
        self.num_envs = env.num_envs * 2

        paired_observation_space = single_observation_space(env)
        assert isinstance(paired_observation_space, Box)
        self.single_observation_space = Box(
            low=paired_observation_space.low[0, :],
            high=paired_observation_space.high[0, :],
            dtype=paired_observation_space.dtype,
        )
        self.observation_space = batch_space(
            self.single_observation_space, n=self.num_envs
        )

        paired_action_space = single_action_space(env)
        assert isinstance(paired_action_space, TupleSpace)
        self.single_action_space = paired_action_space.spaces[0]
        self.action_space = TupleSpace((self.single_action_space,) * self.num_envs)
        self.action_plane_space = MultiDiscrete(ACTION_SIZES)

    def step(self, action: np.ndarray) -> VecEnvStepReturn:
        paired_actions = np.array(np.split(action, len(action) // 2, axis=0))
        obs, p_rew, p_done, p_info = self.env.step(paired_actions)
        return (
            obs,
            np.reshape(p_rew, self.num_envs),
            np.reshape(p_done, self.num_envs),
            [info for pair in p_info for info in pair],
        )
