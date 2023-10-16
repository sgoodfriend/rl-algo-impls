from typing import Optional, Union

import numpy as np

from rl_algo_impls.wrappers.vector_wrapper import VectorEnv, VectorWrapper, find_wrapper


class IncompleteArrayError(Exception):
    pass


class SingleActionMaskWrapper(VectorWrapper):
    def get_action_mask(self) -> Optional[np.ndarray]:
        envs = getattr(self.env.unwrapped, "envs", None)  # type: ignore
        assert (
            envs
        ), f"{self.__class__.__name__} expects to wrap synchronous vectorized env"
        masks = [getattr(e.unwrapped, "action_mask", None) for e in envs]
        assert all(m is not None for m in masks)
        return np.array(masks, dtype=np.bool_)


class MicrortsMaskWrapper(VectorWrapper):
    def get_action_mask(self) -> np.ndarray:
        return self.env.get_action_mask().astype(bool)  # type: ignore


def find_action_masker(
    env: VectorEnv,
) -> Optional[Union[SingleActionMaskWrapper, MicrortsMaskWrapper]]:
    return find_wrapper(env, SingleActionMaskWrapper) or find_wrapper(
        env, MicrortsMaskWrapper
    )
