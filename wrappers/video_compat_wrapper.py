import gym
import numpy as np

from wrappers.vectorable_wrapper import VecotarableWrapper


class VideoCompatWrapper(VecotarableWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def render(self, mode="human", **kwargs):
        r = super().render(mode=mode, **kwargs)
        if mode == "rgb_array" and isinstance(r, np.ndarray) and r.dtype != np.uint8:
            r = r.astype(np.uint8)
        return r
