import numpy as np
from gym import Env

from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnvStepReturn,
    VectorableWrapper,
)


class ScoreRewardWrapper(VectorableWrapper):
    def __init__(
        self, env: Env, delta_every_step: bool = False, episode_end: bool = False
    ) -> None:
        super().__init__(env)
        assert (
            delta_every_step or episode_end
        ), "Either delta_every_step or episode_end must be true"
        self.delta_every_step = delta_every_step
        self.episode_end = episode_end

    def step(self, action) -> VecEnvStepReturn:
        o, r, d, infos = super().step(action)
        r_to_add = []
        if self.delta_every_step:
            r_to_add.append(
                np.expand_dims(
                    np.array([info["score_reward"]["delta_reward"] for info in infos]),
                    axis=-1,
                )
            )
        if self.episode_end:
            r_to_add.append(
                np.expand_dims(
                    np.array(
                        [
                            info.get("results", {}).get("score_reward", 0)
                            for info in infos
                        ]
                    ),
                    axis=-1,
                )
            )
        if len(r.shape) == 1:
            r = np.expand_dims(r, axis=-1)
        rewards = np.concatenate([r] + r_to_add, axis=-1)
        return o, rewards, d, infos
