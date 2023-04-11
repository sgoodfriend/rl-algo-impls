from typing import Dict, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.distributions import Distribution, constraints

from rl_algo_impls.shared.actor import Actor, PiForward, pi_forward
from rl_algo_impls.shared.actor.categorical import MaskedCategorical
from rl_algo_impls.shared.encoder import EncoderOutDim
from rl_algo_impls.shared.module.utils import mlp


class GridnetDistribution(Distribution):
    def __init__(
        self,
        map_size: int,
        action_vec: NDArray[np.int64],
        logits: torch.Tensor,
        masks: torch.Tensor,
        validate_args: Optional[bool] = None,
    ) -> None:
        self.map_size = map_size
        self.action_vec = action_vec

        masks = masks.view(-1, masks.shape[-1])
        split_masks = torch.split(masks, action_vec.tolist(), dim=1)

        grid_logits = logits.reshape(-1, action_vec.sum())
        split_logits = torch.split(grid_logits, action_vec.tolist(), dim=1)
        self.categoricals = [
            MaskedCategorical(logits=lg, validate_args=validate_args, mask=m)
            for lg, m in zip(split_logits, split_masks)
        ]

        batch_shape = logits.size()[:-1] if logits.ndimension() > 1 else torch.Size()
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        prob_stack = torch.stack(
            [
                c.log_prob(a)
                for a, c in zip(action.view(-1, action.shape[-1]).T, self.categoricals)
            ],
            dim=-1,
        )
        logprob = prob_stack.view(-1, self.map_size, len(self.action_vec))
        return logprob.sum(dim=(1, 2))

    def entropy(self) -> torch.Tensor:
        ent = torch.stack([c.entropy() for c in self.categoricals], dim=-1)
        ent = ent.view(-1, self.map_size, len(self.action_vec))
        return ent.sum(dim=(1, 2))

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        s = torch.stack([c.sample(sample_shape) for c in self.categoricals], dim=-1)
        return s.view(-1, self.map_size, len(self.action_vec))

    @property
    def mode(self) -> torch.Tensor:
        m = torch.stack([c.mode for c in self.categoricals], dim=-1)
        return m.view(-1, self.map_size, len(self.action_vec))

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        # Constraints handled by child distributions in dist
        return {}


class GridnetActorHead(Actor):
    def __init__(
        self,
        map_size: int,
        action_vec: NDArray[np.int64],
        in_dim: EncoderOutDim,
        hidden_sizes: Tuple[int, ...] = (32,),
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: bool = True,
    ) -> None:
        super().__init__()
        self.map_size = map_size
        self.action_vec = action_vec
        assert isinstance(in_dim, int)
        layer_sizes = (in_dim,) + hidden_sizes + (map_size * action_vec.sum(),)
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
        assert (
            action_masks is not None
        ), f"No mask case unhandled in {self.__class__.__name__}"
        logits = self._fc(obs)
        pi = GridnetDistribution(self.map_size, self.action_vec, logits, action_masks)
        return pi_forward(pi, actions)

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return (self.map_size, len(self.action_vec))
