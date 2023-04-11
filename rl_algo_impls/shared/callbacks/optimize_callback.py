import numpy as np
import optuna

from time import perf_counter
from torch.utils.tensorboard.writer import SummaryWriter
from typing import NamedTuple, Union

from rl_algo_impls.shared.callbacks import Callback
from rl_algo_impls.shared.callbacks.eval_callback import evaluate
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.wrappers.episode_stats_writer import EpisodeStatsWriter
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv, find_wrapper


class Evaluation(NamedTuple):
    eval_stat: EpisodesStats
    train_stat: EpisodesStats
    score: float


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
        self.last_score = -np.inf

    def on_step(self, timesteps_elapsed: int = 1) -> bool:
        super().on_step(timesteps_elapsed)
        if self.timesteps_elapsed >= self.eval_step * self.step_freq:
            self.evaluate()
            return not self.is_pruned
        return True

    def evaluate(self) -> None:
        self.last_eval_stat, self.last_train_stat, score = evaluation(
            self.policy,
            self.env,
            self.tb_writer,
            self.n_episodes,
            self.deterministic,
            self.timesteps_elapsed,
        )
        self.last_score = score

        self.trial.report(score, self.eval_step)
        if self.trial.should_prune():
            self.is_pruned = True

        self.eval_step += 1


def evaluation(
    policy: Policy,
    env: VecEnv,
    tb_writer: SummaryWriter,
    n_episodes: int,
    deterministic: bool,
    timesteps_elapsed: int,
) -> Evaluation:
    start_time = perf_counter()
    eval_stat = evaluate(
        env,
        policy,
        n_episodes,
        deterministic=deterministic,
        print_returns=False,
    )
    end_time = perf_counter()
    tb_writer.add_scalar(
        "eval/steps_per_second",
        eval_stat.length.sum() / (end_time - start_time),
        timesteps_elapsed,
    )
    policy.train()
    print(f"Eval Timesteps: {timesteps_elapsed} | {eval_stat}")
    eval_stat.write_to_tensorboard(tb_writer, "eval", timesteps_elapsed)

    stats_writer = find_wrapper(policy.env, EpisodeStatsWriter)
    assert stats_writer

    train_stat = EpisodesStats(stats_writer.episodes)
    print(f"  Train Stat: {train_stat}")

    score = (eval_stat.score.mean + train_stat.score.mean) / 2
    print(f"  Score: {round(score, 2)}")
    tb_writer.add_scalar(
        "eval/score",
        score,
        timesteps_elapsed,
    )

    return Evaluation(eval_stat, train_stat, score)
