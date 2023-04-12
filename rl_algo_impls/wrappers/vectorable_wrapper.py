from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from gym import Env, Space, Wrapper
from stable_baselines3.common.vec_env import VecEnv as SB3VecEnv

VecEnvObs = Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]
VecEnvStepReturn = Tuple[VecEnvObs, np.ndarray, np.ndarray, List[Dict]]


class VectorableWrapper(Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.single_observation_space = single_observation_space(env)
        self.single_action_space = single_action_space(env)

    def step(self, action) -> VecEnvStepReturn:
        return self.env.step(action)

    def reset(self) -> VecEnvObs:
        return self.env.reset()


VecEnv = Union[VectorableWrapper, SB3VecEnv]


def single_observation_space(env: Union[VecEnv, Env]) -> Space:
    return getattr(env, "single_observation_space", env.observation_space)


def single_action_space(env: Union[VecEnv, Env]) -> Space:
    return getattr(env, "single_action_space", env.action_space)


W = TypeVar("W", bound=Wrapper)


def find_wrapper(env: VecEnv, wrapper_class: Type[W]) -> Optional[W]:
    current = env
    while current and current != current.unwrapped:
        if isinstance(current, wrapper_class):
            return current
        current = getattr(current, "env")
    return None
