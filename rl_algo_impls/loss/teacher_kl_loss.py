from typing import Dict, Optional

import torch
from torch.nn.modules.loss import _Loss

from rl_algo_impls.runner.config import Config


def teacher_kl_loss_enabled(config: Config) -> bool:
    return bool(config.algo_hyperparams.get("teacher_kl_loss_coef", None))


class TeacherKLLoss(_Loss):
    def __init__(
        self,
        unbiased: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.unbiased = unbiased
        assert (
            self.reduction == "mean"
        ), f"reduction must be 'mean', got {self.reduction}"

    def forward(
        self,
        training_logprobs: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        logratio = teacher_logprobs - training_logprobs
        if self.unbiased:
            ratio = torch.exp(logratio)
            loss = (ratio - 1) - logratio
        else:
            loss = 0.5 * logratio**2
        if weights is not None:
            loss *= weights
        return loss.mean()
