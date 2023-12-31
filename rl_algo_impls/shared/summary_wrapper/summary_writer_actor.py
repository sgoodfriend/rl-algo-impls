import inspect

import ray
from torch.utils.tensorboard.writer import SummaryWriter


@ray.remote
class SummaryWriterActor:
    def __init__(self, tensorboard_summary_path: str):
        self.tb_writer = SummaryWriter(tensorboard_summary_path)
        self.timesteps_elapsed = 0

    def close(self) -> None:
        self.tb_writer.close()

    def call(self, name: str, *args, **kwargs):
        attr = getattr(self.tb_writer, name)
        if "global_step" not in inspect.signature(attr).parameters:
            kwargs.pop("global_step", None)
        return attr(*args, **kwargs)
