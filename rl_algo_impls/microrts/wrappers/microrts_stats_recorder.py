from typing import Any, Dict, List, Optional

import numpy as np

from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnvObs,
    VecEnvStepReturn,
    VectorableWrapper,
)


class MicrortsStatsRecorder(VectorableWrapper):
    def __init__(
        self,
        env,
        bots: Optional[Dict[str, int]] = None,
        map_paths: Optional[List[str]] = None,
    ) -> None:
        super().__init__(env)
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        self.bots = bots
        if self.bots:
            self._bot_at_index = [None] * (env.num_envs - sum(self.bots.values()))
            for b, n in self.bots.items():
                self._bot_at_index.extend([b] * n)

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
                # ScoreRewardFunction makes no sense to accumulate
                info["microrts_stats"] = dict(
                    (n, r)
                    for n, r in zip(raw_names, raw_rewards)
                    if n != "ScoreRewardFunction"
                )

                winloss = raw_rewards[raw_names.index("RAIWinLossRewardFunction")]
                microrts_results = {
                    "win": int(winloss == 1),
                    "draw": int(winloss == 0),
                    "loss": int(winloss == -1),
                }
                bot = self.bot_at_index(idx)
                map_name = self.map_names[idx]
                if bot or map_name:
                    paired_name = "_".join(s for s in [bot, map_name] if s)
                    suffixes = {s for s in [bot, map_name, paired_name] if s}
                    microrts_results.update(
                        {
                            f"{k}_{suffix}": v
                            for k, v in microrts_results.items()
                            for suffix in suffixes
                        }
                    )

                info["microrts_results"] = microrts_results

                self.raw_rewards[idx] = []

    def bot_at_index(self, idx: int) -> Optional[str]:
        if not self.bots:
            return None
        bots = self._bot_at_index
        if hasattr(self.env, "learner_indexes"):
            bots = [b for i, b in zip(self.env.learner_indexes(), bots) if i]
        return self._bot_at_index[idx]
