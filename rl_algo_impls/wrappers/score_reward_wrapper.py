import numpy as np

from rl_algo_impls.wrappers.vector_wrapper import (
    VecEnvStepReturn,
    VectorEnv,
    VectorWrapper,
    get_infos,
)


class ScoreRewardWrapper(VectorWrapper):
    def __init__(
        self, env: VectorEnv, delta_every_step: bool = False, episode_end: bool = False
    ) -> None:
        super().__init__(env)
        assert (
            delta_every_step or episode_end
        ), "Either delta_every_step or episode_end must be true"
        self.delta_every_step = delta_every_step
        self.episode_end = episode_end

    def step(self, action) -> VecEnvStepReturn:
        o, r, terminations, truncations, infos = super().step(action)
        r_to_add = []
        if self.delta_every_step:
            r_to_add.append(
                np.expand_dims(
                    np.array(
                        [
                            sr["delta_reward"]
                            for sr in get_infos(
                                infos, "score_reward", self.env.num_envs, None
                            )
                        ]
                    ),
                    axis=-1,
                )
            )
        if self.episode_end:
            r_to_add.append(
                np.expand_dims(
                    np.array(
                        [
                            r.get("score_reward", 0) if r is not None else 0
                            for r in get_infos(
                                infos, "results", self.env.num_envs, None
                            )
                        ]
                    ),
                    axis=-1,
                )
            )
        if len(r.shape) == 1:
            r = np.expand_dims(r, axis=-1)
        rewards = np.concatenate([r] + r_to_add, axis=-1)
        return o, rewards, terminations, truncations, infos
