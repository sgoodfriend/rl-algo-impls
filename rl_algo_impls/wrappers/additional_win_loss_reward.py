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
        rewards = np.column_stack([r, winloss])
        return o, rewards, d, infos
