import inspect

from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)


class InProcessSummaryWrapper(AbstractSummaryWrapper):
    def __init__(self, tensorboard_summary_path: str):
        self.tb_writer = SummaryWriter(tensorboard_summary_path)
        self.timesteps_elapsed = 0

    def on_timesteps_elapsed(self, timesteps_elapsed: int) -> None:
        self.timesteps_elapsed = timesteps_elapsed

    def close(self) -> None:
        self.tb_writer.close()

    def __getattr__(self, name: str):
        attr = getattr(self.tb_writer, name)
        if callable(attr) and "global_step" in inspect.signature(attr).parameters:

            def wrapper(*args, **kwargs):
                attr(*args, global_step=self.timesteps_elapsed, **kwargs)

            return wrapper
        else:
            return attr
