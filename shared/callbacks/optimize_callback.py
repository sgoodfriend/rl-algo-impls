import numpy as np
import optuna

from time import perf_counter
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Union

from shared.callbacks.callback import Callback
from shared.callbacks.eval_callback import evaluate
from shared.policy.policy import Policy
from shared.stats import EpisodesStats
from wrappers.episode_stats_writer import EpisodeStatsWriter
from wrappers.vectorable_wrapper import VecEnv, find_wrapper


class OptimizeCallback(Callback):
    def __init__(
        self,
        policy: Policy,
        env: VecEnv,
        trial: optuna.Trial,
        tb_writer: SummaryWriter,
        step_freq: Union[int, float] = 50_000,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.env = env
        self.trial = trial
        self.tb_writer = tb_writer
        self.step_freq = step_freq
        self.n_episodes = n_episodes
        self.deterministic = deterministic

        stats_writer = find_wrapper(policy.env, EpisodeStatsWriter)
        assert stats_writer
        self.stats_writer = stats_writer

        self.eval_step = 1
        self.is_pruned = False
        self.last_eval_stat = None
        self.last_train_stat = None

    def on_step(self, timesteps_elapsed: int = 1) -> bool:
        super().on_step(timesteps_elapsed)
        if self.timesteps_elapsed >= self.eval_step * self.step_freq:
            self.evaluate()
            return not self.is_pruned
        return True

    def evaluate(self) -> EpisodesStats:
        start_time = perf_counter()
        eval_stat = evaluate(
            self.env,
            self.policy,
            self.n_episodes,
            deterministic=self.deterministic,
            print_returns=False,
        )
        end_time = perf_counter()
        self.tb_writer.add_scalar(
            "eval/steps_per_second",
            eval_stat.length.sum() / (end_time - start_time),
            self.timesteps_elapsed,
        )
        self.policy.train()
        print(f"Eval Timesteps: {self.timesteps_elapsed} | {eval_stat}")
        eval_stat.write_to_tensorboard(self.tb_writer, "eval", self.timesteps_elapsed)

        train_stat = EpisodesStats(self.stats_writer.episodes)
        print(f"  Train Stat: {train_stat}")

        self.last_eval_stat = eval_stat
        self.last_train_stat = train_stat

        score = self.last_score()
        print(f"  Score: {round(score, 2)}")
        self.tb_writer.add_scalar(
            "eval/score",
            score,
            self.timesteps_elapsed,
        )

        self.trial.report(score, self.eval_step)
        if self.trial.should_prune():
            self.is_pruned = True

        self.eval_step += 1

        return eval_stat

    def last_score(self) -> float:
        eval_score = self.last_eval_stat.score.mean if self.last_eval_stat else np.NINF
        train_score = (
            self.last_train_stat.score.mean if self.last_train_stat else np.NINF
        )
        return (eval_score + train_score) / 2
