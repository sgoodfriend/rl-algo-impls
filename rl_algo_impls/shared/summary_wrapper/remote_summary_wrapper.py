import logging
from typing import Any, Dict

import ray

from rl_algo_impls.runner.config import Config, TrainArgs
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.shared.summary_wrapper.summary_writer_actor import SummaryWriterActor


class RemoteSummaryWrapper(AbstractSummaryWrapper):
    def __init__(self, config: Config, args: TrainArgs):
        self.tb_writer_actor = SummaryWriterActor.remote(config, args)
        self.timesteps_elapsed = 0

        self.maybe_add_logging_handler(logging.getLogger())

    def on_timesteps_elapsed(self, timesteps_elapsed: int) -> None:
        self.timesteps_elapsed = timesteps_elapsed

    def close(self) -> None:
        ray.get(self.tb_writer_actor.close.remote())

    def __getattr__(self, name: str):
        def method(*args, **kwargs):
            self.tb_writer_actor.call.remote(
                name, *args, global_step=self.timesteps_elapsed, **kwargs
            )

        return method

    def update_summary(self, summary_update: Dict[str, Any]) -> None:
        self.tb_writer_actor.update_summary.remote(summary_update)

    def make_wandb_archive(self, path: str) -> None:
        self.tb_writer_actor.make_wandb_archive.remote(path)

    def log_video(self, video_path: str, fps: int) -> None:
        self.tb_writer_actor.log_video.remote(video_path, fps, self.timesteps_elapsed)

    def maybe_add_logging_handler(self, logger: logging.Logger) -> None:
        logger.addHandler(RemoteSummaryWrapperLoggingHandler(self))


class RemoteSummaryWrapperLoggingHandler(logging.Handler):
    def __init__(self, tb_writer: RemoteSummaryWrapper):
        super().__init__()
        self.tb_writer = tb_writer

    def emit(self, record: logging.LogRecord) -> None:
        self.tb_writer.tb_writer_actor.log_text.remote(
            record.levelno, self.format(record), self.tb_writer.timesteps_elapsed
        )
