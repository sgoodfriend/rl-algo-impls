from typing import Dict, List, Union

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter


class TrainStats:
    def __init__(
        self,
        step_stats: List[Dict[str, Union[float, np.ndarray]]],
        explained_var: float,
    ) -> None:
        self.step_stats: Dict[str, Union[float, np.ndarray]] = {}
        for k, item in step_stats[0].items():
            if isinstance(item, np.ndarray):
                self.step_stats[k] = np.mean([s[k] for s in step_stats], axis=0)
            else:
                self.step_stats[k] = np.mean([s[k] for s in step_stats]).item()
        self.explained_var = explained_var

    def write_to_tensorboard(self, tb_writer: SummaryWriter, global_step: int) -> None:
        stats = {**self.step_stats, **{"explained_var": self.explained_var}}
        for name, value in stats.items():
            if isinstance(value, np.ndarray):
                for idx, v in enumerate(value.flatten()):
                    tb_writer.add_scalar(
                        f"losses/{name}_{idx}", v, global_step=global_step
                    )
            else:
                tb_writer.add_scalar(f"losses/{name}", value, global_step=global_step)

    def __repr__(self) -> str:
        stats = {**self.step_stats, **{"explained_var": self.explained_var}}
        s = []
        for name, value in stats.items():
            if isinstance(value, np.ndarray):
                s.append(f"{name}: " + ",".join(round(v, 2) for v in value.flatten()))
            else:
                s.append(f"{name}: {round(value, 2)}")
        return " | ".join(s)
