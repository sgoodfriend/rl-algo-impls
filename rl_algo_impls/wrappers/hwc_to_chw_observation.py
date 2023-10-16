from typing import Tuple

import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.experimental.vector.vector_env import VectorObservationWrapper
from gymnasium.spaces import Box

from rl_algo_impls.wrappers.vector_wrapper import ObsType, VectorEnv


class HwcToChwObservation(ObservationWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

        assert isinstance(env.observation_space, Box)
        self.observation_space, self._transpose_axes = transpose_space(
            env.observation_space
        )
        if hasattr(env, "single_observation_space"):
            self.single_observation_space, _ = transpose_space(
                getattr(env, "single_observation_space")
            )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        full_shape = obs.shape
        obs_shape = self.observation_space.shape
        addl_dims = len(full_shape) - len(obs_shape)
        if addl_dims > 0:
            transpose_axes = list(range(addl_dims))
            transpose_axes.extend(a + addl_dims for a in self._transpose_axes)
        else:
            transpose_axes = self._transpose_axes
        return np.transpose(obs, axes=transpose_axes)


class HwcToChwVectorObservation(VectorObservationWrapper):
    def __init__(self, env: VectorEnv) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, Box)
        assert isinstance(env.single_observation_space, Box)
        self.observation_space, self._transpose_axes = transpose_space(
            env.observation_space
        )
        (
            self.single_observation_space,
            self._transpose_single_obs_axes,
        ) = transpose_space(env.single_observation_space)

    def vector_observation(self, observation: ObsType) -> ObsType:
        assert isinstance(observation, np.ndarray)
        return np.transpose(observation, axes=self._transpose_axes)

    def single_observation(self, observation: ObsType) -> ObsType:
        assert isinstance(observation, np.ndarray)
        return np.transpose(observation, axes=self._transpose_single_obs_axes)


def transpose_space(space: Box) -> Tuple[Box, Tuple[int, ...]]:
    axes = tuple(i for i in range(len(space.shape)))
    transpose_axes = axes[:-3] + (axes[-1],) + axes[-3:-1]

    transposed_space = Box(
        low=np.transpose(space.low, axes=transpose_axes),
        high=np.transpose(space.high, axes=transpose_axes),
        shape=[space.shape[idx] for idx in transpose_axes],
        dtype=space.dtype,
    )

    return transposed_space, transpose_axes
