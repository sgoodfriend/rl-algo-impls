from typing import Dict, Optional

import torch
from torch.nn.modules.loss import _Loss

from rl_algo_impls.rollout.rollout import Batch
from rl_algo_impls.shared.policy.abstract_policy import AbstractPolicy


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

    def add_to_batch(
        self, teacher: AbstractPolicy, batch: Batch
    ) -> Dict[str, torch.Tensor]:
        teacher.reset_noise()
        with torch.no_grad():
            return {
                "teacher_logprobs": teacher(
                    batch.obs, batch.actions, action_masks=batch.action_masks
                ).logp_a,
            }

    def forward(
        self,
        training_logprobs: torch.Tensor,
        mb_additional: Dict[str, torch.Tensor],
        weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        teacher_logprobs = mb_additional["teacher_logprobs"]
        logratio = teacher_logprobs - training_logprobs
        if self.unbiased:
            ratio = torch.exp(logratio)
            loss = (ratio - 1) - logratio
        else:
            loss = 0.5 * logratio**2
        if weights is not None:
            loss *= weights
        return loss.mean()
