from typing import Optional

import numpy as np
from gym import Env

from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnvStepReturn,
    VectorableWrapper,
)


class AdditionalWinLossRewardWrapper(VectorableWrapper):
    def __init__(
        self, env: Env, label_smoothing_factor: Optional[float] = None
    ) -> None:
        super().__init__(env)
        self.label_smoothing_factor = label_smoothing_factor

    def step(self, action) -> VecEnvStepReturn:
        o, r, d, infos = super().step(action)
        winloss = np.array(
            [info.get("results", {}).get("WinLoss", 0) for info in infos],
            dtype=np.float32,
        )
        if self.label_smoothing_factor is not None:
            winloss *= self.label_smoothing_factor
        if len(r.shape) == 1:
            r = np.expand_dims(r, axis=-1)
        rewards = np.concatenate([r, np.expand_dims(winloss, axis=-1)], axis=-1)
        return o, rewards, d, infos
