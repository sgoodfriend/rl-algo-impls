from typing import Tuple, TypeVar

import gymnasium
import numpy as np
from numpy.typing import NDArray

from rl_algo_impls.wrappers.vector_wrapper import VectorEnv, VectorWrapper

RunningMeanStdSelf = TypeVar("RunningMeanStdSelf", bound="RunningMeanStd")


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

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


class NormalizeObservation(VectorWrapper):
    def __init__(
        self,
        env: VectorEnv,
        training: bool = True,
        epsilon: float = 1e-8,
        clip: float = 10.0,
    ) -> None:
        super().__init__(env)
        self.rms = RunningMeanStd(shape=env.single_observation_space.shape)
        self.training = training
        self.epsilon = epsilon
        self.clip = clip

    def step(self, action):
        obs, reward, terminations, truncations, info = self.env.step(action)
        return self.normalize(obs), reward, terminations, truncations, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.normalize(obs), info

    def normalize(self, obs: NDArray) -> NDArray:
        if self.training:
            self.rms.update(obs)
        normalized = np.clip(
            (obs - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
            -self.clip,
            self.clip,
        )
        return normalized

    def save(self, path: str) -> None:
        self.rms.save(path)

    def load(self, path: str) -> None:
        self.rms.load(path)

    def load_from(
        self: NormalizeObservationSelf, existing: NormalizeObservationSelf
    ) -> None:
        self.rms.load_from(existing.rms)


NormalizeRewardSelf = TypeVar("NormalizeRewardSelf", bound="NormalizeReward")


class NormalizeReward(VectorWrapper):
    def __init__(
        self,
        env: VectorEnv,
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
        obs, reward, terminations, truncations, info = self.env.step(action)

        reward = self.normalize(reward)

        self.returns[terminations | truncations] = 0

        return obs, reward, terminations, truncations, info

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
