import gym
import numpy as np

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from typing import Union

ObsType = Union[np.ndarray, dict]
ActType = Union[int, float, np.ndarray, dict]


class InitialStepOffsetWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, initial_steps_to_offset: int) -> None:
        super().__init__(env)
        self.initial_steps_to_offset = initial_steps_to_offset
        self.initialized = False

    def reset(self, **kwargs) -> ObsType:
        obs = super().reset(**kwargs)
        if not self.initialized:
            for _ in range(self.initial_steps_to_offset):
                obs, _, _, _ = self.env.step(self.env.action_space.sample())
            self.initialized = True
        return obs
