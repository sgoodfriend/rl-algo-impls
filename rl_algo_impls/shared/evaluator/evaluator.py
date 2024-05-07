import itertools
import logging
import os
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np

from rl_algo_impls.runner.config import Config
from rl_algo_impls.runner.env_hyperparams import EnvHyperparams
from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.data_store.data_store_data import EvalView
from rl_algo_impls.shared.data_store.data_store_view import EvalDataStoreView
from rl_algo_impls.shared.policy.abstract_policy import AbstractPolicy
from rl_algo_impls.shared.stats import Episode, EpisodeAccumulator, EpisodesStats
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.shared.tensor_utils import batch_dict_keys
from rl_algo_impls.shared.vec_env.make_env import make_eval_env
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper
from rl_algo_impls.wrappers.vec_episode_recorder import VecEpisodeRecorder
from rl_algo_impls.wrappers.vector_wrapper import VectorEnv


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
            logging.info(
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
    env: VectorEnv,
    policy: AbstractPolicy,
    n_episodes: int,
    render: bool = False,
    deterministic: bool = True,
    print_returns: bool = True,
    ignore_first_episode: bool = False,
    additional_keys_to_log: Optional[List[str]] = None,
    score_function: str = "mean-std",
) -> EpisodesStats:
    policy.eval()
    policy.reset_noise()

    episodes = EvaluateAccumulator(
        env.num_envs,
        n_episodes,
        print_returns,
        ignore_first_episode,
        additional_keys_to_log=additional_keys_to_log,
    )

    obs, _ = env.reset()
    get_action_mask = getattr(env, "get_action_mask", None)

    while not episodes.is_done():
        act = policy.act(
            obs,
            deterministic=deterministic,
            action_masks=(
                batch_dict_keys(get_action_mask()) if get_action_mask else None
            ),
        )
        obs, rew, terminations, truncations, info = env.step(act)

        done = terminations | truncations
        episodes.step(rew, done, info)
        if render:
            env.render()
    stats = EpisodesStats(
        episodes.episodes,
        score_function=score_function,
    )
    if print_returns:
        logging.info(stats)
    return stats


class Evaluator:
    def __init__(
        self,
        config: Config,
        data_store_accessor: AbstractDataStoreAccessor,
        tb_writer: AbstractSummaryWrapper,
        self_play_wrapper: Optional[SelfPlayWrapper] = None,
        best_model_path: Optional[str] = None,
        n_episodes: int = 10,
        save_best: bool = True,
        deterministic: bool = True,
        only_record_video_on_best: bool = True,
        video_dir: Optional[str] = None,
        max_video_length: int = 9000,
        ignore_first_episode: bool = False,
        additional_keys_to_log: Optional[List[str]] = None,
        score_function: str = "mean-std",
        wandb_enabled: bool = False,
        score_threshold: Optional[float] = None,
        only_checkpoint_best_policies: bool = False,
        latest_model_path: Optional[str] = None,
        disable_video_generation: bool = False,
    ) -> None:
        super().__init__()
        self.data_store_view = EvalDataStoreView(data_store_accessor)
        self.env = make_eval_env(
            config,
            EnvHyperparams(**config.env_hyperparams),
            self.data_store_view,
            self_play_wrapper=self_play_wrapper,
        )
        self.tb_writer = tb_writer
        self.best_model_path = best_model_path
        self.n_episodes = n_episodes
        self.save_best = save_best
        self.deterministic = deterministic
        self.stats: List[EpisodesStats] = []
        self.best = None

        self.only_record_video_on_best = only_record_video_on_best
        self.max_video_length = max_video_length
        self.video_dir = video_dir
        if video_dir and not disable_video_generation:
            os.makedirs(video_dir, exist_ok=True)
            self.video_env = VecEpisodeRecorder(
                make_eval_env(
                    config,
                    EnvHyperparams(**config.env_hyperparams),
                    self.data_store_view,
                    override_hparams={"n_envs": 1},
                    self_play_wrapper=self_play_wrapper,
                ),
                video_dir,  # This is updated when a video is actually created
                max_video_length=self.max_video_length,
                tb_writer=tb_writer,
            )
        else:
            self.video_env = None

        self.ignore_first_episode = ignore_first_episode
        self.additional_keys_to_log = additional_keys_to_log
        self.score_function = score_function
        self.wandb_enabled = wandb_enabled
        self.score_threshold = score_threshold
        self.latest_model_path = latest_model_path

        self.only_checkpoint_best_policies = only_checkpoint_best_policies

    def evaluate(
        self,
        eval_data: EvalView,
        n_episodes: Optional[int] = None,
        print_returns: Optional[bool] = None,
    ) -> EpisodesStats:
        start_time = perf_counter()
        policy, self.timesteps_elapsed = self.data_store_view.update_from_eval_data(
            eval_data
        )
        self.tb_writer.on_timesteps_elapsed(self.timesteps_elapsed)
        eval_stat = evaluate(
            self.env,
            policy,
            n_episodes or self.n_episodes,
            deterministic=self.deterministic,
            print_returns=print_returns or False,
            ignore_first_episode=self.ignore_first_episode,
            additional_keys_to_log=self.additional_keys_to_log,
            score_function=self.score_function,
        )
        end_time = perf_counter()
        self.tb_writer.add_scalar(
            "eval/steps_per_second",
            eval_stat.length.sum() / (end_time - start_time),
        )
        policy.train(True)
        logging.info(f"Eval Timesteps: {self.timesteps_elapsed} | {eval_stat}")

        self.stats.append(eval_stat)

        if self.score_threshold is not None:
            is_best = eval_stat.score.score() >= self.score_threshold
            strictly_better = eval_stat.score.score() > self.score_threshold
        else:
            is_best = not self.best or eval_stat >= self.best
            strictly_better = not self.best or eval_stat > self.best

        if self.latest_model_path:
            self.save(policy, self.latest_model_path)
        if is_best:
            self.best = eval_stat
            if self.save_best:
                assert self.best_model_path
                self.save(policy, self.best_model_path)
                logging.info("Saved best model")
                self.tb_writer.make_wandb_archive(self.best_model_path)

            self.best.write_to_tensorboard(self.tb_writer, "best_eval")
        if self.video_env and (not self.only_record_video_on_best or strictly_better):
            self.generate_video(policy)

        eval_stat.write_to_tensorboard(self.tb_writer, "eval")
        self.checkpoint_policy(policy, is_best)
        return eval_stat

    def checkpoint_policy(self, policy: AbstractPolicy, is_best: bool):
        if self.only_checkpoint_best_policies:
            if not is_best:
                return
            else:
                logging.info(f"Checkpointing best policy at {self.timesteps_elapsed}")
        self.data_store_view.submit_checkpoint(policy)

    def generate_video(self, policy: AbstractPolicy) -> None:
        assert self.video_env and self.video_dir
        best_video_base_path = os.path.join(self.video_dir, str(self.timesteps_elapsed))
        self.video_env.base_path = best_video_base_path
        video_stats = evaluate(
            self.video_env,
            policy,
            1,
            deterministic=self.deterministic,
            print_returns=False,
            score_function=self.score_function,
        )
        logging.info(f"Saved video: {video_stats}")

    def save(self, policy: AbstractPolicy, model_path: str) -> None:
        self.data_store_view.save(policy, model_path)
