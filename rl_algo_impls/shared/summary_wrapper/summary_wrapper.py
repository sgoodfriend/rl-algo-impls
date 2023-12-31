import inspect

from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.shared.callbacks.callback import Callback


class SummaryWrapper:
    def __init__(self, tb_writer: SummaryWriter):
        self.tb_writer = tb_writer
        self.timesteps_elapsed = 0

    def on_steps(self, timesteps_elapsed: int) -> None:
        self.timesteps_elapsed += timesteps_elapsed

    def __getattr__(self, name: str):
        attr = getattr(self.tb_writer, name)
        if callable(attr) and "global_step" in inspect.signature(attr).parameters:

            def wrapper(*args, **kwargs):
                attr(*args, global_step=self.timesteps_elapsed, **kwargs)

            return wrapper
        else:
            return attr
