import ray

from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.shared.summary_wrapper.summary_writer_actor import SummaryWriterActor


class RemoteSummaryWrapper(AbstractSummaryWrapper):
    def __init__(self, tensorboard_summary_path: str):
        self.tb_writer_actor = SummaryWriterActor.remote(tensorboard_summary_path)
        self.timesteps_elapsed = 0

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
