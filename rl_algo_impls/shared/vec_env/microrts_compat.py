from typing import TypeVar

from gym_microrts.envs.vec_env import (
    MicroRTSGridModeSharedMemVecEnv,
    MicroRTSGridModeVecEnv,
)

MicroRTSGridModeVecEnvCompatSelf = TypeVar(
    "MicroRTSGridModeVecEnvCompatSelf", bound="MicroRTSGridModeVecEnvCompat"
)


class MicroRTSGridModeVecEnvCompat(MicroRTSGridModeVecEnv):
    @property
    def unwrapped(
        self: MicroRTSGridModeVecEnvCompatSelf,
    ) -> MicroRTSGridModeVecEnvCompatSelf:
        return self


MicroRTSGridModeSharedMemVecEnvCompatSelf = TypeVar(
    "MicroRTSGridModeSharedMemVecEnvCompatSelf",
    bound="MicroRTSGridModeSharedMemVecEnvCompat",
)


class MicroRTSGridModeSharedMemVecEnvCompat(MicroRTSGridModeSharedMemVecEnv):
    @property
    def unwrapped(
        self: MicroRTSGridModeSharedMemVecEnvCompatSelf,
    ) -> MicroRTSGridModeSharedMemVecEnvCompatSelf:
        return self
