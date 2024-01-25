from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional

import numpy as np

from rl_algo_impls.ppo.appo_train_stats import APPOTrainStats
from rl_algo_impls.ppo.ppo import TrainStats


class DPPPOTrainStepStats(NamedTuple):
    loss: float
    pi_loss: float
    v_loss: np.ndarray
    entropy_loss: float
    approx_kl: float
    clipped_frac: float
    val_clipped_frac: np.ndarray
    additional_losses: Dict[str, float]
    grad_norm: Optional[float]


@dataclass
class DPPPOTrainStats(APPOTrainStats):
    def __init__(
        self,
        step_stats: List[DPPPOTrainStepStats],
        explained_var: float,
        n_epochs: float,
    ) -> None:
        self.loss = np.mean([s.loss for s in step_stats]).item()
        self.pi_loss = np.mean([s.pi_loss for s in step_stats]).item()
        self.v_loss = np.mean([s.v_loss for s in step_stats], axis=0)
        self.entropy_loss = np.mean([s.entropy_loss for s in step_stats]).item()
        self.approx_kl = np.mean([s.approx_kl for s in step_stats]).item()
        self.clipped_frac = np.mean([s.clipped_frac for s in step_stats]).item()
        self.val_clipped_frac = np.mean(
            [s.val_clipped_frac for s in step_stats], axis=0
        )
        self.additional_losses = {
            k: np.mean([s.additional_losses[k] for s in step_stats]).item()
            for k in step_stats[0].additional_losses
        }
        self.explained_var = explained_var
        self.grad_norm = np.mean(
            [s.grad_norm for s in step_stats if s.grad_norm is not None]
        ).item()
        self.n_epochs = n_epochs
