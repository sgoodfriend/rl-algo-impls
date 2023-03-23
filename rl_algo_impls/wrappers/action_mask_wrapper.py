import numpy as np

from gym.vector.vector_env import VectorEnv
from stable_baselines3.common.vec_env import VecEnv as SBVecEnv
from typing import Optional

from rl_algo_impls.wrappers.vectorable_wrapper import VecotarableWrapper


class IncompleteArrayError(Exception):
    pass


class ActionMaskWrapper(VecotarableWrapper):
    def action_masks(self) -> Optional[np.ndarray]:
        envs = getattr(self.env.unwrapped, "envs")
        assert (
            envs
        ), f"{self.__class__.__name__} expects to wrap synchronous vectorized env"
        masks = [getattr(e.unwrapped, "action_mask") for e in envs]
        assert all(m is not None for m in masks)
        return np.array(masks, dtype=np.bool8)
