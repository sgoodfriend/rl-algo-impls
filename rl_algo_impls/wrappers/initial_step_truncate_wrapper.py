import logging
from typing import Any, Dict, SupportsFloat, Tuple, Union

import gymnasium
import numpy as np

ObsType = Union[np.ndarray, dict]
ActType = Union[int, float, np.ndarray, dict]


class InitialStepTruncateWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, initial_steps_to_truncate: int) -> None:
        super().__init__(env)
        self.initial_steps_to_truncate = initial_steps_to_truncate
        self.initialized = initial_steps_to_truncate == 0
        self.steps = 0

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, rew, terminated, truncated, info = self.env.step(action)
        if not self.initialized:
            self.steps += 1
            if self.steps >= self.initial_steps_to_truncate:
                logging.info(f"Truncation at {self.steps} steps")
                truncated = True
                self.initialized = True
        return obs, rew, terminated, truncated, info
