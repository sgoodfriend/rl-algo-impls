from typing import Optional

import numpy as np
from gym.vector.sync_vector_env import SyncVectorEnv
from stable_baselines3.common.vec_env.base_vec_env import tile_images

from rl_algo_impls.wrappers.vectorable_wrapper import VectorableWrapper


class SyncVectorEnvRenderCompat(VectorableWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        base_env = self.env.unwrapped
        if isinstance(base_env, SyncVectorEnv):
            imgs = [env.render(mode="rgb_array") for env in base_env.envs]
            bigimg = tile_images(imgs)
            if mode == "human":
                import cv2

                cv2.imshow("vecenv", bigimg[:, :, ::-1])
                cv2.waitKey(1)
            elif mode == "rgb_array":
                return bigimg
            else:
                raise NotImplemented(f"Render mode {mode} is not supported")
        else:
            return self.env.render(mode=mode)
