import gym
import numpy as np

from typing import Union

ObsType = Union[np.ndarray, dict]
ActType = Union[int, float, np.ndarray, dict]


class NoRewardTimeout(gym.Wrapper):

    def __init__(self, env: gym.Env, n_timeout_steps: int) -> None:
        super().__init__(env)
        self.n_timeout_steps = n_timeout_steps
        self.steps_since_reward = 0

        self.episode_score = 0
        self.episode_step_idx = 0

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        obs, rew, done, info = self.env.step(action)

        self.episode_score += rew
        self.episode_step_idx += 1

        if rew > 0 or done:
            self.steps_since_reward = 0
        else:
            self.steps_since_reward += 1
            if self.steps_since_reward >= self.n_timeout_steps:
                print(f"{self.__class__.__name__}: Early terminate | "
                      f"Score: {self.episode_score} | "
                      f"Length: {self.episode_step_idx}")
                done = True
                self.steps_since_reward = 0

        if done:
            self.episode_score = 0
            self.episode_step_idx = 0

        return obs, rew, done, info
