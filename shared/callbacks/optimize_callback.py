import optuna

from time import perf_counter
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Union

from shared.callbacks.callback import Callback
from shared.callbacks.eval_callback import evaluate
from shared.policy.policy import Policy
from shared.stats import EpisodesStats
from wrappers.vectorable_wrapper import VecEnv


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

        self.eval_step = 1
        self.is_pruned = False
        self.last_eval_stat = None

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

        self.trial.report(eval_stat.score.mean, self.eval_step)
        if self.trial.should_prune():
            self.is_pruned = True

        self.eval_step += 1
        self.last_eval_stat = eval_stat

        return eval_stat
