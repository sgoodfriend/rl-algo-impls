import gymnasium
import numpy as np


class VideoCompatWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)

    def render(self):
        r = super().render()
        if isinstance(r, np.ndarray) and r.dtype != np.uint8:
            r = r.astype(np.uint8)
        return r
