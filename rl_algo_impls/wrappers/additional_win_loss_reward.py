import numpy as np

from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnvStepReturn,
    VectorableWrapper,
)


class AdditionalWinLossRewardWrapper(VectorableWrapper):
    def step(self, action) -> VecEnvStepReturn:
        o, r, d, infos = super().step(action)
        winloss = np.array(
            [info.get("results", {}).get("WinLoss", 0) for info in infos]
        )
        if len(r.shape) == 1:
            r = np.expand_dims(r, axis=-1)
        rewards = np.concatenate([r, np.expand_dims(winloss, axis=-1)], axis=-1)
        return o, rewards, d, infos
