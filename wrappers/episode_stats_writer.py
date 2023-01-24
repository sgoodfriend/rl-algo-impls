import gym
import numpy as np

from collections import deque
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvStepReturn,
    VecEnvWrapper,
    VecEnvObs,
)
from torch.utils.tensorboard.writer import SummaryWriter

from shared.stats import Episode, EpisodesStats


class EpisodeStatsWriter(VecEnvWrapper):
    def __init__(
        self, venv, tb_writer: SummaryWriter, training: bool = True, rolling_length=100
    ):
        super().__init__(venv)
        self.training = training
        self.tb_writer = tb_writer
        self.rolling_length = rolling_length
        self.episodes = deque(maxlen=rolling_length)
        self.total_steps = 0
        self.episode_cnt = 0
        self.last_episode_cnt_print = 0

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()
        self.total_steps += self.venv.num_envs
        step_episodes = []
        for info in infos:
            ep_info = info.get("episode")
            if ep_info:
                episode = Episode(ep_info["r"], ep_info["l"])
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
        return obs, rews, dones, infos

    def reset(self) -> VecEnvObs:
        return self.venv.reset()
