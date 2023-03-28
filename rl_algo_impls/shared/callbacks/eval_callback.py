import itertools
import os
from time import perf_counter
from typing import Dict, List, Optional, Union

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.shared.callbacks.callback import Callback
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.stats import Episode, EpisodeAccumulator, EpisodesStats
from rl_algo_impls.wrappers.action_mask_wrapper import find_action_masker
from rl_algo_impls.wrappers.vec_episode_recorder import VecEpisodeRecorder
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv


class EvaluateAccumulator(EpisodeAccumulator):
    def __init__(
        self,
        num_envs: int,
        goal_episodes: int,
        print_returns: bool = True,
        ignore_first_episode: bool = False,
        additional_keys_to_log: Optional[List[str]] = None,
    ):
        super().__init__(num_envs)
        self.completed_episodes_by_env_idx = [[] for _ in range(num_envs)]
        self.goal_episodes_per_env = int(np.ceil(goal_episodes / num_envs))
        self.print_returns = print_returns
        if ignore_first_episode:
            first_done = set()

            def should_record_done(idx: int) -> bool:
                has_done_first_episode = idx in first_done
                first_done.add(idx)
                return has_done_first_episode

            self.should_record_done = should_record_done
        else:
            self.should_record_done = lambda idx: True
        self.additional_keys_to_log = additional_keys_to_log

    def on_done(self, ep_idx: int, episode: Episode, info: Dict) -> None:
        if self.additional_keys_to_log:
            episode.info = {k: info[k] for k in self.additional_keys_to_log}
        if (
            self.should_record_done(ep_idx)
            and len(self.completed_episodes_by_env_idx[ep_idx])
            >= self.goal_episodes_per_env
        ):
            return
        self.completed_episodes_by_env_idx[ep_idx].append(episode)
        if self.print_returns:
            print(
                f"Episode {len(self)} | "
                f"Score {episode.score} | "
                f"Length {episode.length}"
            )

    def __len__(self) -> int:
        return sum(len(ce) for ce in self.completed_episodes_by_env_idx)

    @property
    def episodes(self) -> List[Episode]:
        return list(itertools.chain(*self.completed_episodes_by_env_idx))

    def is_done(self) -> bool:
        return all(
            len(ce) == self.goal_episodes_per_env
            for ce in self.completed_episodes_by_env_idx
        )


def evaluate(
    env: VecEnv,
    policy: Policy,
    n_episodes: int,
    render: bool = False,
    deterministic: bool = True,
    print_returns: bool = True,
    ignore_first_episode: bool = False,
    additional_keys_to_log: Optional[List[str]] = None,
) -> EpisodesStats:
    policy.sync_normalization(env)
    policy.eval()

    episodes = EvaluateAccumulator(
        env.num_envs,
        n_episodes,
        print_returns,
        ignore_first_episode,
        additional_keys_to_log=additional_keys_to_log,
    )

    obs = env.reset()
    action_masker = find_action_masker(env)
    while not episodes.is_done():
        act = policy.act(
            obs,
            deterministic=deterministic,
            action_masks=action_masker.action_masks() if action_masker else None,
        )
        obs, rew, done, info = env.step(act)
        episodes.step(rew, done, info)
        if render:
            env.render()
    stats = EpisodesStats(episodes.episodes)
    if print_returns:
        print(stats)
    return stats


class EvalCallback(Callback):
    def __init__(
        self,
        policy: Policy,
        env: VecEnv,
        tb_writer: SummaryWriter,
        best_model_path: Optional[str] = None,
        step_freq: Union[int, float] = 50_000,
        n_episodes: int = 10,
        save_best: bool = True,
        deterministic: bool = True,
        record_best_videos: bool = True,
        video_env: Optional[VecEnv] = None,
        best_video_dir: Optional[str] = None,
        max_video_length: int = 3600,
        ignore_first_episode: bool = False,
        additional_keys_to_log: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.env = env
        self.tb_writer = tb_writer
        self.best_model_path = best_model_path
        self.step_freq = int(step_freq)
        self.n_episodes = n_episodes
        self.save_best = save_best
        self.deterministic = deterministic
        self.stats: List[EpisodesStats] = []
        self.best = None

        self.record_best_videos = record_best_videos
        assert video_env or not record_best_videos
        self.video_env = video_env
        assert best_video_dir or not record_best_videos
        self.best_video_dir = best_video_dir
        if best_video_dir:
            os.makedirs(best_video_dir, exist_ok=True)
        self.max_video_length = max_video_length
        self.best_video_base_path = None
        self.ignore_first_episode = ignore_first_episode
        self.additional_keys_to_log = additional_keys_to_log

    def on_step(self, timesteps_elapsed: int = 1) -> bool:
        super().on_step(timesteps_elapsed)
        if self.timesteps_elapsed // self.step_freq >= len(self.stats):
            self.evaluate()
        return True

    def evaluate(
        self, n_episodes: Optional[int] = None, print_returns: Optional[bool] = None
    ) -> EpisodesStats:
        start_time = perf_counter()
        eval_stat = evaluate(
            self.env,
            self.policy,
            n_episodes or self.n_episodes,
            deterministic=self.deterministic,
            print_returns=print_returns or False,
            ignore_first_episode=self.ignore_first_episode,
            additional_keys_to_log=self.additional_keys_to_log,
        )
        end_time = perf_counter()
        self.tb_writer.add_scalar(
            "eval/steps_per_second",
            eval_stat.length.sum() / (end_time - start_time),
            self.timesteps_elapsed,
        )
        self.policy.train(True)
        print(f"Eval Timesteps: {self.timesteps_elapsed} | {eval_stat}")

        self.stats.append(eval_stat)

        if not self.best or eval_stat >= self.best:
            strictly_better = not self.best or eval_stat > self.best
            self.best = eval_stat
            if self.save_best:
                assert self.best_model_path
                self.policy.save(self.best_model_path)
                print("Saved best model")
            self.best.write_to_tensorboard(
                self.tb_writer, "best_eval", self.timesteps_elapsed
            )
            if strictly_better and self.record_best_videos:
                assert self.video_env and self.best_video_dir
                self.best_video_base_path = os.path.join(
                    self.best_video_dir, str(self.timesteps_elapsed)
                )
                video_wrapped = VecEpisodeRecorder(
                    self.video_env,
                    self.best_video_base_path,
                    max_video_length=self.max_video_length,
                )
                video_stats = evaluate(
                    video_wrapped,
                    self.policy,
                    1,
                    deterministic=self.deterministic,
                    print_returns=False,
                )
                print(f"Saved best video: {video_stats}")

        eval_stat.write_to_tensorboard(self.tb_writer, "eval", self.timesteps_elapsed)

        return eval_stat
