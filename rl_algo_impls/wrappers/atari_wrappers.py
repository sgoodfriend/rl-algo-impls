from typing import Any, Dict

import gymnasium
import numpy as np


class EpisodicLifeEnv(gymnasium.Wrapper):
    def __init__(
        self, env: gymnasium.Env, training: bool = True, noop_act: int = 0
    ) -> None:
        super().__init__(env)
        self.training = training
        self.noop_act = noop_act
        self.life_done_continue = False
        self.lives = 0

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        new_lives = self.env.unwrapped.ale.lives()
        self.life_done_continue = new_lives != self.lives and not (
            terminated or truncated
        )
        # Only if training should life-end be marked as done
        if self.training and 0 < new_lives < self.lives:
            truncated = True
        self.lives = new_lives
        return obs, rew, terminated, truncated, info

    def reset(self, **kwargs):
        # If life_done_continue (but not game over), then a reset should just allow the
        # game to progress to the next life.
        if self.training and self.life_done_continue:
            obs, _, _, _, info = self.env.step(self.noop_act)
        else:
            obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireOnLifeStarttEnv(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, fire_act: int = 1) -> None:
        super().__init__(env)
        self.fire_act = fire_act
        action_meanings = env.unwrapped.get_action_meanings()
        assert action_meanings[fire_act] == "FIRE"
        assert len(action_meanings) >= 3
        self.lives = 0
        self.fire_on_next_step = True

    def step(self, action):
        if self.fire_on_next_step:
            action = self.fire_act
            self.fire_on_next_step = False
        obs, rew, terminated, truncated, info = self.env.step(action)
        new_lives = self.env.unwrapped.ale.lives()
        done = terminated or truncated
        if 0 < new_lives < self.lives and not done:
            self.fire_on_next_step = True
        self.lives = new_lives
        return obs, rew, terminated, truncated, info

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(self.fire_act)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        self.fire_on_next_step = False
        return obs, info


class ClipRewardEnv(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, training: bool = True) -> None:
        super().__init__(env)
        self.training = training

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if self.training:
            info["unclipped_reward"] = rew
            rew = np.sign(rew)  # type: ignore
        return obs, rew, terminated, truncated, info
