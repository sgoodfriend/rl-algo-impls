from collections import deque
from typing import List, Optional

from rl_algo_impls.shared.stats import Episode, EpisodesStats
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.wrappers.vector_wrapper import (
    VecEnvResetReturn,
    VecEnvStepReturn,
    VectorEnv,
    VectorWrapper,
    get_info,
    get_infos,
)


class EpisodeStatsWriter(VectorWrapper):
    def __init__(
        self,
        env: VectorEnv,
        tb_writer: AbstractSummaryWrapper,
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
        self._record_stats_enabled = True

    def step(self, actions) -> VecEnvStepReturn:
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self._record_stats(infos)
        return obs, rewards, terminations, truncations, infos

    def reset(self, **kwargs) -> VecEnvResetReturn:
        obs, infos = self.env.reset(**kwargs)
        self._record_stats(infos)
        return obs, infos

    @property
    def steps_per_step(self) -> int:
        if self._steps_per_step is None:
            return getattr(self.env, "num_envs", 1)
        return self._steps_per_step

    @steps_per_step.setter
    def steps_per_step(self, steps_per_step: int) -> None:
        self._steps_per_step = steps_per_step

    def disable_record_stats(self) -> None:
        self._record_stats_enabled = False

    def enable_record_stats(self) -> None:
        self._record_stats_enabled = True

    def _record_stats(self, infos: dict) -> None:
        if not self._record_stats_enabled:
            return
        self.total_steps += self.steps_per_step
        step_episodes = []
        if "episode" not in infos:
            # look for episode in final_info
            for env_idx, final_info in enumerate(
                get_infos(infos, "final_info", self.num_envs, {})
            ):
                if final_info and "episode" in final_info:
                    additional_info = {
                        k: final_info[k] for k in self.additional_keys_to_log
                    }
                    episode = Episode(
                        final_info["episode"]["r"].item(),
                        final_info["episode"]["l"].item(),
                        info=additional_info,
                    )
                    step_episodes.append(episode)
                    self.episodes.append(episode)
        else:
            for env_idx, ep_info in enumerate(
                get_infos(infos, "episode", self.num_envs, {})
            ):
                if ep_info:
                    additional_info = {
                        k: get_info(infos, k, env_idx)
                        for k in self.additional_keys_to_log
                    }
                    episode = Episode(ep_info["r"], ep_info["l"], info=additional_info)
                    step_episodes.append(episode)
                    self.episodes.append(episode)

        if step_episodes:
            tag = "train" if self.training else "eval"
            step_stats = EpisodesStats(step_episodes, simple=True)
            step_stats.write_to_tensorboard(self.tb_writer, tag)
            rolling_stats = EpisodesStats(self.episodes)
            rolling_stats.write_to_tensorboard(self.tb_writer, f"{tag}_rolling")
            self.episode_cnt += len(step_episodes)
            if self.episode_cnt >= self.last_episode_cnt_print + self.rolling_length:
                print(
                    f"Episode: {self.episode_cnt} | "
                    f"Steps: {self.total_steps} | "
                    f"{rolling_stats}"
                )
                self.last_episode_cnt_print += self.rolling_length
