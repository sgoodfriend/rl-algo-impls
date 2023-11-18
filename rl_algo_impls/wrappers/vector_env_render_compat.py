from typing import Optional

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import tile_images

from rl_algo_impls.wrappers.vector_wrapper import VectorEnv, VectorWrapper


class VectorEnvRenderCompat(VectorWrapper):
    def __init__(self, env: VectorEnv) -> None:
        super().__init__(env)
        try:
            self.render_mode = self.env.call("render_mode")[0]  # type: ignore
        except AttributeError:
            self.render_mode = "rgb_array"

    def render(self) -> Optional[np.ndarray]:
        imgs = [img for img in self.env.call("render") if img is not None]  # type: ignore
        if imgs:
            return tile_images(imgs)
