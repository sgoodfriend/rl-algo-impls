from dataclasses import asdict, dataclass
from typing import Dict, List, Union

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter


@dataclass
class TrainStepStats:
    loss: float
    pi_loss: float
    v_loss: np.ndarray
    entropy_loss: float


class TrainStats:
    data: Dict[str, Union[float, np.ndarray]]

    def __init__(self, step_stats: List[TrainStepStats], explain_var: float) -> None:
        self.data = {"explained_var": explain_var}
        for k in asdict(step_stats[0]).keys():
            if isinstance(getattr(step_stats[0], k), np.ndarray):
                self.data[k] = np.mean([getattr(s, k) for s in step_stats], axis=0)
            else:
                self.data[k] = np.mean([getattr(s, k) for s in step_stats]).item()

    def write_to_tensorboard(self, tb_writer: SummaryWriter, global_step: int) -> None:
        for name, value in self.data.items():
            if isinstance(value, np.ndarray):
                for idx, v in enumerate(value.flatten()):
                    tb_writer.add_scalar(
                        f"losses/{name}_{idx}", v, global_step=global_step
                    )
            else:
                tb_writer.add_scalar(f"losses/{name}", value, global_step=global_step)
