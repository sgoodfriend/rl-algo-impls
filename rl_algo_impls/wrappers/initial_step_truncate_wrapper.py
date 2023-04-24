from typing import Any, Dict, Tuple, Union

import gym
import numpy as np

from rl_algo_impls.wrappers.vectorable_wrapper import VectorableWrapper

ObsType = Union[np.ndarray, dict]
ActType = Union[int, float, np.ndarray, dict]


class InitialStepTruncateWrapper(VectorableWrapper):
    def __init__(self, env: gym.Env, initial_steps_to_truncate: int) -> None:
        super().__init__(env)
        self.initial_steps_to_truncate = initial_steps_to_truncate
        self.initialized = initial_steps_to_truncate == 0
        self.steps = 0

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, Dict[str, Any]]:
        obs, rew, done, info = self.env.step(action)
        if not self.initialized:
            self.steps += 1
            if self.steps >= self.initial_steps_to_truncate:
                print(f"Truncation at {self.steps} steps")
                done = True
                self.initialized = True
        return obs, rew, done, info
