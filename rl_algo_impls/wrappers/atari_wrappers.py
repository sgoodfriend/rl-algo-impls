from typing import Any, Dict, Tuple, Union

import gym
import numpy as np

from rl_algo_impls.wrappers.vectorable_wrapper import VectorableWrapper

ObsType = Union[np.ndarray, dict]
ActType = Union[int, float, np.ndarray, dict]


class EpisodicLifeEnv(VectorableWrapper):
    def __init__(self, env: gym.Env, training: bool = True, noop_act: int = 0) -> None:
        super().__init__(env)
        self.training = training
        self.noop_act = noop_act
        self.life_done_continue = False
        self.lives = 0

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, Dict[str, Any]]:
        obs, rew, done, info = self.env.step(action)
        new_lives = self.env.unwrapped.ale.lives()
        self.life_done_continue = new_lives != self.lives and not done
        # Only if training should life-end be marked as done
        if self.training and 0 < new_lives < self.lives:
            done = True
        self.lives = new_lives
        return obs, rew, done, info

    def reset(self, **kwargs) -> ObsType:
        # If life_done_continue (but not game over), then a reset should just allow the
        # game to progress to the next life.
        if self.training and self.life_done_continue:
            obs, _, _, _ = self.env.step(self.noop_act)
        else:
            obs = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FireOnLifeStarttEnv(VectorableWrapper):
    def __init__(self, env: gym.Env, fire_act: int = 1) -> None:
        super().__init__(env)
        self.fire_act = fire_act
        action_meanings = env.unwrapped.get_action_meanings()
        assert action_meanings[fire_act] == "FIRE"
        assert len(action_meanings) >= 3
        self.lives = 0
        self.fire_on_next_step = True

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, Dict[str, Any]]:
        if self.fire_on_next_step:
            action = self.fire_act
            self.fire_on_next_step = False
        obs, rew, done, info = self.env.step(action)
        new_lives = self.env.unwrapped.ale.lives()
        if 0 < new_lives < self.lives and not done:
            self.fire_on_next_step = True
        self.lives = new_lives
        return obs, rew, done, info

    def reset(self, **kwargs) -> ObsType:
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(self.fire_act)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        self.fire_on_next_step = False
        return obs


class ClipRewardEnv(VectorableWrapper):
    def __init__(self, env: gym.Env, training: bool = True) -> None:
        super().__init__(env)
        self.training = training

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, Dict[str, Any]]:
        obs, rew, done, info = self.env.step(action)
        if self.training:
            info["unclipped_reward"] = rew
            rew = np.sign(rew)
        return obs, rew, done, info
