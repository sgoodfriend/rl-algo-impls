from typing import Tuple, TypeVar

import gym
import numpy as np
from numpy.typing import NDArray

from rl_algo_impls.wrappers.vectorable_wrapper import (
    VectorableWrapper,
    single_observation_space,
)

RunningMeanStdSelf = TypeVar("RunningMeanStdSelf", bound="RunningMeanStd")


class RunningMeanStd:
    def __init__(self, episilon: float = 1e-4, shape: Tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = episilon

    def update(self, x: NDArray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            mean=self.mean,
            var=self.var,
            count=self.count,
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        self.mean = data["mean"]
        self.var = data["var"]
        self.count = data["count"]

    def load_from(self: RunningMeanStdSelf, existing: RunningMeanStdSelf) -> None:
        self.mean = np.copy(existing.mean)
        self.var = np.copy(existing.var)
        self.count = np.copy(existing.count)


NormalizeObservationSelf = TypeVar(
    "NormalizeObservationSelf", bound="NormalizeObservation"
)


class NormalizeObservation(VectorableWrapper):
    def __init__(
        self,
        env: gym.Env,
        training: bool = True,
        epsilon: float = 1e-8,
        clip: float = 10.0,
    ) -> None:
        super().__init__(env)
        self.rms = RunningMeanStd(shape=single_observation_space(env).shape)
        self.training = training
        self.epsilon = epsilon
        self.clip = clip

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.normalize(obs), reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.normalize(obs)

    def normalize(self, obs: NDArray) -> NDArray:
        obs_array = np.array([obs]) if not self.is_vector_env else obs
        if self.training:
            self.rms.update(obs_array)
        normalized = np.clip(
            (obs_array - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
            -self.clip,
            self.clip,
        )
        return normalized[0] if not self.is_vector_env else normalized

    def save(self, path: str) -> None:
        self.rms.save(path)

    def load(self, path: str) -> None:
        self.rms.load(path)

    def load_from(
        self: NormalizeObservationSelf, existing: NormalizeObservationSelf
    ) -> None:
        self.rms.load_from(existing.rms)


NormalizeRewardSelf = TypeVar("NormalizeRewardSelf", bound="NormalizeReward")


class NormalizeReward(VectorableWrapper):
    def __init__(
        self,
        env: gym.Env,
        training: bool = True,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip: float = 10.0,
    ) -> None:
        super().__init__(env)
        self.rms = RunningMeanStd(shape=())
        self.training = training
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip

        self.returns = np.zeros(self.num_envs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if not self.is_vector_env:
            reward = np.array([reward])
        reward = self.normalize(reward)
        if not self.is_vector_env:
            reward = reward[0]

        dones = done if self.is_vector_env else np.array([done])
        self.returns[dones] = 0

        return obs, reward, done, info

    def reset(self, **kwargs):
        self.returns = np.zeros(self.num_envs)
        return self.env.reset(**kwargs)

    def normalize(self, rewards):
        if self.training:
            self.returns = self.returns * self.gamma + rewards
            self.rms.update(self.returns)
        return np.clip(
            rewards / np.sqrt(self.rms.var + self.epsilon), -self.clip, self.clip
        )

    def save(self, path: str) -> None:
        self.rms.save(path)

    def load(self, path: str) -> None:
        self.rms.load(path)

    def load_from(self: NormalizeRewardSelf, existing: NormalizeRewardSelf) -> None:
        self.rms.load_from(existing.rms)
