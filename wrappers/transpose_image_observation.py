import gym
import numpy as np

from gym import ObservationWrapper
from gym.spaces import Box


class TransposeImageObservation(ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        assert isinstance(env.observation_space, Box)

        obs_space = env.observation_space
        axes = tuple(i for i in range(len(obs_space.shape)))
        self._transpose_axes = axes[:-3] + (axes[-1],) + axes[-3:-1]

        self.observation_space = Box(
            low=np.transpose(obs_space.low, axes=self._transpose_axes),
            high=np.transpose(obs_space.high, axes=self._transpose_axes),
            shape=[obs_space.shape[idx] for idx in self._transpose_axes],
            dtype=obs_space.dtype,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.transpose(obs, axes=self._transpose_axes)
