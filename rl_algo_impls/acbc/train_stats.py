from typing import Dict, List

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter


class TrainStats:
    def __init__(
        self, step_stats: List[Dict[str, float]], explained_var: float
    ) -> None:
        self.step_stats = {
            k: np.mean([s[k] for s in step_stats]).item() for k in step_stats[0]
        }
        self.explained_var = explained_var

    def write_to_tensorboard(self, tb_writer: SummaryWriter, global_step: int) -> None:
        stats = {**self.step_stats, **{"explained_var": self.explained_var}}
        for name, value in stats.items():
            tb_writer.add_scalar(f"losses/{name}", value, global_step=global_step)

    def __repr__(self) -> str:
        stats = {**self.step_stats, **{"explained_var": self.explained_var}}
        return " | ".join(f"{k}: {round(v, 2)}" for k, v in stats.items())
