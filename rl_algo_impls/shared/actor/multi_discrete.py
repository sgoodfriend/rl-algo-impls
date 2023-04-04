from typing import Dict, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.distributions import Distribution, constraints

from rl_algo_impls.shared.actor.actor import Actor, PiForward, pi_forward
from rl_algo_impls.shared.actor.categorical import MaskedCategorical
from rl_algo_impls.shared.encoder import EncoderOutDim
from rl_algo_impls.shared.module.utils import mlp


class MultiCategorical(Distribution):
    def __init__(
        self,
        nvec: NDArray[np.int64],
        probs=None,
        logits=None,
        validate_args=None,
        masks: Optional[torch.Tensor] = None,
    ):
        # Either probs or logits should be set
        assert (probs is None) != (logits is None)
        masks_split = (
            torch.split(masks, nvec.tolist(), dim=1)
            if masks is not None
            else [None] * len(nvec)
        )
        if probs:
            self.dists = [
                MaskedCategorical(probs=p, validate_args=validate_args, mask=m)
                for p, m in zip(torch.split(probs, nvec.tolist(), dim=1), masks_split)
            ]
            param = probs
        else:
            assert logits is not None
            self.dists = [
                MaskedCategorical(logits=lg, validate_args=validate_args, mask=m)
                for lg, m in zip(torch.split(logits, nvec.tolist(), dim=1), masks_split)
            ]
            param = logits
        batch_shape = param.size()[:-1] if param.ndimension() > 1 else torch.Size()
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        prob_stack = torch.stack(
            [c.log_prob(a) for a, c in zip(action.T, self.dists)], dim=-1
        )
        return prob_stack.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([c.entropy() for c in self.dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return torch.stack([c.sample(sample_shape) for c in self.dists], dim=-1)

    @property
    def mode(self) -> torch.Tensor:
        return torch.stack([c.mode for c in self.dists], dim=-1)

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        # Constraints handled by child distributions in dist
        return {}


class MultiDiscreteActorHead(Actor):
    def __init__(
        self,
        nvec: NDArray[np.int64],
        in_dim: EncoderOutDim,
        hidden_sizes: Tuple[int, ...] = (32,),
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: bool = True,
    ) -> None:
        super().__init__()
        self.nvec = nvec
        assert isinstance(in_dim, int)
        layer_sizes = (in_dim,) + hidden_sizes + (nvec.sum(),)
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
        pi = MultiCategorical(self.nvec, logits=logits, masks=action_masks)
        return pi_forward(pi, actions)

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return (len(self.nvec),)
