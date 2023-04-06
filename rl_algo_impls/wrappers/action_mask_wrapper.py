from typing import Optional, Union

import numpy as np
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnv,
    VecotarableWrapper,
    find_wrapper,
)


class IncompleteArrayError(Exception):
    pass


class SingleActionMaskWrapper(VecotarableWrapper):
    def action_masks(self) -> Optional[np.ndarray]:
        envs = getattr(self.env.unwrapped, "envs")  # type: ignore
        assert (
            envs
        ), f"{self.__class__.__name__} expects to wrap synchronous vectorized env"
        masks = [getattr(e.unwrapped, "action_mask") for e in envs]
        assert all(m is not None for m in masks)
        return np.array(masks, dtype=np.bool_)


class MicrortsMaskWrapper(VecotarableWrapper):
    def action_masks(self) -> np.ndarray:
        microrts_env = self.env.unwrapped  # type: ignore
        assert isinstance(microrts_env, MicroRTSGridModeVecEnv)
        return microrts_env.get_action_mask().astype(bool)


def find_action_masker(
    env: VecEnv,
) -> Optional[Union[SingleActionMaskWrapper, MicrortsMaskWrapper]]:
    return find_wrapper(env, SingleActionMaskWrapper) or find_wrapper(
        env, MicrortsMaskWrapper
    )
