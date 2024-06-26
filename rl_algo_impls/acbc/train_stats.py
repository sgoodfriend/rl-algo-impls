from typing import Dict, List, Union

import numpy as np

from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)


class TrainStats:
    def __init__(
        self,
        step_stats: List[Dict[str, Union[float, np.ndarray]]],
        explained_var: float,
        grad_norms: List[float],
    ) -> None:
        self.step_stats: Dict[str, Union[float, np.ndarray]] = {}
        for k, item in step_stats[0].items():
            if isinstance(item, np.ndarray):
                self.step_stats[k] = np.mean([s[k] for s in step_stats], axis=0)
            else:
                self.step_stats[k] = np.mean([s[k] for s in step_stats]).item()
        self.explained_var = explained_var
        self.grad_norm = np.mean(grad_norms).item()

    def write_to_tensorboard(self, tb_writer: AbstractSummaryWrapper) -> None:
        stats = {
            **self.step_stats,
            **{"explained_var": self.explained_var, "grad_norm": self.grad_norm},
        }
        for name, value in stats.items():
            if isinstance(value, np.ndarray):
                for idx, v in enumerate(value.flatten()):
                    tb_writer.add_scalar(f"losses/{name}_{idx}", v)
            else:
                tb_writer.add_scalar(f"losses/{name}", value)

    def __repr__(self) -> str:
        stats = {
            **self.step_stats,
            **{"explained_var": self.explained_var, "grad_norm": self.grad_norm},
        }
        s = []
        for name, value in stats.items():
            if isinstance(value, np.ndarray):
                s.append(f"{name}: " + ",".join(round(v, 2) for v in value.flatten()))
            else:
                s.append(f"{name}: {round(value, 2)}")
        return " | ".join(s)
