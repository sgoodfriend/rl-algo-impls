from typing import Any, Dict, List

import numpy as np

from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnvObs,
    VecEnvStepReturn,
    VecotarableWrapper,
)


class MicrortsStatsRecorder(VecotarableWrapper):
    def __init__(self, env, gamma: float) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.raw_rewards = [[] for _ in range(self.num_envs)]

    def reset(self) -> VecEnvObs:
        obs = super().reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.env.step(actions)
        self._update_infos(infos, dones)
        return obs, rews, dones, infos

    def _update_infos(self, infos: List[Dict[str, Any]], dones: np.ndarray) -> None:
        for idx, info in enumerate(infos):
            self.raw_rewards[idx].append(info["raw_rewards"])
        for idx, (info, done) in enumerate(zip(infos, dones)):
            if done:
                raw_rewards = np.array(self.raw_rewards[idx]).sum(0)
                raw_names = [str(rf) for rf in self.env.unwrapped.rfs]
                info["microrts_stats"] = dict(zip(raw_names, raw_rewards))
                self.raw_rewards[idx] = []
