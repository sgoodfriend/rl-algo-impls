from typing import Any, Dict, List, Optional

import numpy as np

from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnvObs,
    VecEnvStepReturn,
    VectorableWrapper,
)


class MicrortsStatsRecorder(VectorableWrapper):
    def __init__(
        self, env, gamma: float, bots: Optional[Dict[str, int]] = None
    ) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        self.bots = bots
        if self.bots:
            self.bot_at_index = [None] * (env.num_envs - sum(self.bots.values()))
            for b, n in self.bots.items():
                self.bot_at_index.extend([b] * n)
        else:
            self.bot_at_index = [None] * env.num_envs

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

                winloss = raw_rewards[raw_names.index("WinLossRewardFunction")]
                microrts_results = {
                    "win": int(winloss == 1),
                    "draw": int(winloss == 0),
                    "loss": int(winloss == -1),
                }
                bot = self.bot_at_index[idx]
                if bot:
                    microrts_results.update(
                        {f"{k}_{bot}": v for k, v in microrts_results.items()}
                    )

                info["microrts_results"] = microrts_results

                self.raw_rewards[idx] = []
