import gym
import numpy as np

from rl_algo_impls.wrappers.vectorable_wrapper import VectorableWrapper


class VideoCompatWrapper(VectorableWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def render(self, mode="human", **kwargs):
        r = super().render(mode=mode, **kwargs)
        if mode == "rgb_array" and isinstance(r, np.ndarray) and r.dtype != np.uint8:
            r = r.astype(np.uint8)
        return r
