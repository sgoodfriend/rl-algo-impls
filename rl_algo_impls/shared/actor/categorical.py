import logging
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl_algo_impls.shared.actor import Actor, PiForward, pi_forward
from rl_algo_impls.shared.module.utils import mlp


class MaskedCategorical(Categorical):
    def __init__(
        self,
        probs=None,
        logits=None,
        validate_args=None,
        mask: Optional[torch.Tensor] = None,
        verify: bool = False,
        neg_inf: Optional[float] = None,
    ):
        self.verify = verify
        if mask is not None:
            assert logits is not None, "mask requires logits and not probs"
            neg_inf = neg_inf if neg_inf is not None else torch.finfo(logits.dtype).min
            logits = torch.where(mask, logits, neg_inf)
            if self.verify:
                self.rows_with_valid_actions = mask.any(dim=1)
                self.non_empty_action_mask = mask[self.rows_with_valid_actions]
        self.mask = mask
        super().__init__(probs, logits, validate_args)

    def log_prob(self, value):
        logp = super().log_prob(value)
        if self.verify and self.mask is not None:
            valid_actions = self.non_empty_action_mask[
                torch.arange(self.non_empty_action_mask.shape[0]),
                value[self.rows_with_valid_actions],
            ]
            if not (len(valid_actions) == 0 or valid_actions.all()):
                logging.error(f"INVALID ACTIONS SELECTED")
        return logp

    def entropy(self) -> torch.Tensor:
        if self.mask is None:
            return super().entropy()
        # If mask set, then use approximation for entropy
        p_log_p = self.logits * self.probs  # type: ignore
        masked = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -masked.sum(-1)


class CategoricalActorHead(Actor):
    def __init__(
        self,
        act_dim: int,
        in_dim: int,
        hidden_sizes: Tuple[int, ...] = (32,),
        activation: Type[nn.Module] = nn.Tanh,
        init_layers_orthogonal: bool = True,
    ) -> None:
        super().__init__()
        layer_sizes = (in_dim,) + hidden_sizes + (act_dim,)
        self._fc = mlp(
            layer_sizes,
            activation,
            init_layers_orthogonal=init_layers_orthogonal,
            final_layer_gain=0.01,
        )

    def forward(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
    ) -> PiForward:
        logits = self._fc(obs)
        pi = MaskedCategorical(logits=logits, mask=action_masks)
        return pi_forward(pi, actions)
