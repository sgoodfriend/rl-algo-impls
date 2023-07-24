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
from rl_algo_impls.shared.tensor_utils import TensorOrDict


class GridnetDistribution(Distribution):
    def __init__(
        self,
        map_size: int,
        action_vec: NDArray[np.int64],
        logits: torch.Tensor,
        masks: TensorOrDict,
        validate_args: Optional[bool] = None,
        subaction_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self.map_size = map_size
        self.action_vec = action_vec

        masks_per_position = masks["per_position"] if isinstance(masks, dict) else masks
        masks_per_position = masks_per_position.view(-1, masks_per_position.shape[-1])
        split_masks_per_position = torch.split(
            masks_per_position, action_vec.tolist(), dim=1
        )

        grid_logits = logits.reshape(-1, logits.shape[-1])
        grid_logits_per_position = grid_logits[:, : action_vec.sum()]
        split_logits_per_position = torch.split(
            grid_logits_per_position, action_vec.tolist(), dim=1
        )
        self.categoricals_per_position = [
            MaskedCategorical(logits=lg, validate_args=validate_args, mask=m)
            for lg, m in zip(split_logits_per_position, split_masks_per_position)
        ]
        self.subaction_mask = subaction_mask

        if isinstance(masks, dict) and "pick_position" in masks:
            masks_pick_position = masks["pick_position"]
            self.pick_vec = [masks_pick_position.shape[-1]] * masks_pick_position.shape[
                -2
            ]
            masks_pick_position = masks_pick_position.view(
                -1, masks_pick_position.shape[-1]
            )

            split_masks_pick_position = torch.split(
                masks_pick_position,
                self.pick_vec,
                dim=1,
            )
            logits_pick_position = logits[:, :, :, action_vec.sum() :]
            logits_pick_position = logits_pick_position.reshape(
                logits_pick_position.shape[0], -1, logits_pick_position.shape[-1]
            ).transpose(-1, -2)
            logits_pick_position = logits_pick_position.reshape(
                -1, logits_pick_position.shape[-1]
            )
            split_logits_pick_position = torch.split(
                logits_pick_position, self.pick_vec, dim=1
            )
            self.categoricals_pick_position = [
                MaskedCategorical(logits=lg, validate_args=validate_args, mask=m)
                for lg, m in zip(split_logits_pick_position, split_masks_pick_position)
            ]
        else:
            self.categoricals_pick_position = None

        batch_shape = logits.size()[:-1] if logits.ndimension() > 1 else torch.Size()
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, action: TensorOrDict) -> torch.Tensor:
        action_per_position = (
            action["per_position"] if isinstance(action, dict) else action
        )
        action_per_position = action_per_position.view(
            -1, action_per_position.shape[-1]
        ).T

        prob_stack_per_position = torch.stack(
            [
                (
                    torch.where(
                        action_per_position[0] == self.subaction_mask[idx - 1],
                        c.log_prob(a),
                        0,
                    )
                    if idx > 0 and self.subaction_mask is not None
                    else c.log_prob(a)
                )
                for idx, (a, c) in enumerate(
                    zip(
                        action_per_position,
                        self.categoricals_per_position,
                    )
                )
            ],
            dim=-1,
        )
        logprob_per_position = prob_stack_per_position.view(
            -1, self.map_size, len(self.action_vec)
        ).sum(dim=(1, 2))

        if isinstance(action, dict) and "pick_position" in action:
            assert self.categoricals_pick_position is not None
            action_pick_position = action["pick_position"]
            prob_stack_pick_position = torch.stack(
                [
                    c.log_prob(a)
                    for a, c in zip(
                        action_pick_position.view(-1, len(self.pick_vec)).T,
                        self.categoricals_pick_position,
                    )
                ],
                dim=-1,
            )
            logprob_pick_position = prob_stack_pick_position.sum(dim=-1)
            return logprob_per_position + logprob_pick_position

        return logprob_per_position

    def entropy(self) -> torch.Tensor:
        ent_per_position = (
            torch.stack([c.entropy() for c in self.categoricals_per_position], dim=-1)
            .view(-1, self.map_size, len(self.action_vec))
            .sum(dim=(1, 2))
        )
        if self.categoricals_pick_position:
            ent_pick_position = (
                torch.stack(
                    [c.entropy() for c in self.categoricals_pick_position], dim=-1
                )
                .view(-1, len(self.pick_vec))
                .sum(dim=1)
            )
            return ent_per_position + ent_pick_position
        return ent_per_position

    def sample(self, sample_shape: torch.Size = torch.Size()) -> TensorOrDict:
        s_per_position = torch.stack(
            [c.sample(sample_shape) for c in self.categoricals_per_position], dim=-1
        ).view(-1, self.map_size, len(self.action_vec))
        if self.categoricals_pick_position:
            s_pick_position = torch.stack(
                [c.sample(sample_shape) for c in self.categoricals_pick_position],
                dim=-1,
            )
            return {"per_position": s_per_position, "pick_position": s_pick_position}
        return s_per_position

    @property
    def mode(self) -> TensorOrDict:
        m_per_position = torch.stack(
            [c.mode for c in self.categoricals_per_position], dim=-1
        ).view(-1, self.map_size, len(self.action_vec))
        if self.categoricals_pick_position:
            m_pick_position = torch.stack(
                [c.mode for c in self.categoricals_pick_position], dim=-1
            )
            return {"per_position": m_per_position, "pick_position": m_pick_position}
        return m_per_position

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
