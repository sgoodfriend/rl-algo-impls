from typing import TypeVar

from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

MicroRTSGridModeVecEnvCompatSelf = TypeVar(
    "MicroRTSGridModeVecEnvCompatSelf", bound="MicroRTSGridModeVecEnvCompat"
)


class MicroRTSGridModeVecEnvCompat(MicroRTSGridModeVecEnv):
    @property
    def unwrapped(
        self: MicroRTSGridModeVecEnvCompatSelf,
    ) -> MicroRTSGridModeVecEnvCompatSelf:
        return self
