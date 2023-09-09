from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.shared.stats import Episode, EpisodesStats
from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnvObs,
    VecEnvStepReturn,
    VectorableWrapper,
)


class EpisodeStatsWriter(VectorableWrapper):
    def __init__(
        self,
        env,
        tb_writer: SummaryWriter,
        training: bool = True,
        rolling_length=100,
        additional_keys_to_log: Optional[List[str]] = None,
    ):
        super().__init__(env)
        self.training = training
        self.tb_writer = tb_writer
        self.rolling_length = rolling_length
        self.episodes = deque(maxlen=rolling_length)
        self.total_steps = 0
        self.episode_cnt = 0
        self.last_episode_cnt_print = 0
        self.additional_keys_to_log = (
            additional_keys_to_log if additional_keys_to_log is not None else []
        )
        self._steps_per_step: Optional[int] = None

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.env.step(actions)
        self._record_stats(infos)
        return obs, rews, dones, infos

    # Support for stable_baselines3.common.vec_env.VecEnvWrapper
    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.env.step_wait()
        self._record_stats(infos)
        return obs, rews, dones, infos

    @property
    def steps_per_step(self) -> int:
        if self._steps_per_step is None:
            return getattr(self.env, "num_envs", 1)
        return self._steps_per_step

    @steps_per_step.setter
    def steps_per_step(self, steps_per_step: int) -> None:
        self._steps_per_step = steps_per_step

    def _record_stats(self, infos: List[Dict[str, Any]]) -> None:
        self.total_steps += self.steps_per_step
        step_episodes = []
        for info in infos:
            ep_info = info.get("episode")
            if ep_info:
                additional_info = {k: info[k] for k in self.additional_keys_to_log}
                episode = Episode(ep_info["r"], ep_info["l"], info=additional_info)
                step_episodes.append(episode)
                self.episodes.append(episode)
        if step_episodes:
            tag = "train" if self.training else "eval"
            step_stats = EpisodesStats(step_episodes, simple=True)
            step_stats.write_to_tensorboard(self.tb_writer, tag, self.total_steps)
            rolling_stats = EpisodesStats(self.episodes)
            rolling_stats.write_to_tensorboard(
                self.tb_writer, f"{tag}_rolling", self.total_steps
            )
            self.episode_cnt += len(step_episodes)
            if self.episode_cnt >= self.last_episode_cnt_print + self.rolling_length:
                print(
                    f"Episode: {self.episode_cnt} | "
                    f"Steps: {self.total_steps} | "
                    f"{rolling_stats}"
                )
                self.last_episode_cnt_print += self.rolling_length
