from typing import Optional, Union

import numpy as np

from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnv,
    VecotarableWrapper,
    find_wrapper,
)


class IncompleteArrayError(Exception):
    pass


class SingleActionMaskWrapper(VecotarableWrapper):
    def action_masks(self) -> Optional[np.ndarray]:
        envs = getattr(self.env.unwrapped, "envs")
        assert (
            envs
        ), f"{self.__class__.__name__} expects to wrap synchronous vectorized env"
        masks = [getattr(e.unwrapped, "action_mask") for e in envs]
        assert all(m is not None for m in masks)
        return np.array(masks, dtype=np.bool8)


class MicrortsMaskWrapper(VecotarableWrapper):
    def action_masks(self) -> np.ndarray:
        microrts_env = self.env.unwrapped  # type: ignore
        vec_client = getattr(microrts_env, "vec_client")
        assert (
            vec_client
        ), f"{microrts_env.__class__.__name__} must have vec_client property (as MicroRTSVecEnv does)"
        return np.array(vec_client.getMasks(0), dtype=np.bool8)


def find_action_masker(
    env: VecEnv,
) -> Optional[Union[SingleActionMaskWrapper, MicrortsMaskWrapper]]:
    return find_wrapper(env, SingleActionMaskWrapper) or find_wrapper(
        env, MicrortsMaskWrapper
    )
