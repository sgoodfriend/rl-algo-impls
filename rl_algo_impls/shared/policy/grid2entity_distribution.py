from typing import Dict, NamedTuple, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.distributions import Distribution, constraints

from rl_algo_impls.shared.actor import Actor, PiForward, pi_forward
from rl_algo_impls.shared.actor.categorical import MaskedCategorical
from rl_algo_impls.shared.actor.gridnet import ValueDependentMask
from rl_algo_impls.shared.encoder import EncoderOutDim
from rl_algo_impls.shared.module.utils import mlp
from rl_algo_impls.shared.tensor_utils import TensorOrDict

DEBUG_VERIFY = False
if DEBUG_VERIFY:
    import logging


class Grid2EntityDistribution(Distribution):
    def __init__(
        self,
        keep_mask: torch.Tensor,  # Bool[B, H*W]
        n_keep: torch.Tensor,  # Int[B]
        action_vec: NDArray[np.int64],
        action_logits: torch.Tensor,  # Float[B, S, A]
        action_masks: torch.Tensor,  # Bool[B, S, A]
        validate_args: Optional[bool] = None,
        subaction_mask: Optional[Dict[int, ValueDependentMask]] = None,
        entropy_mask_correction: bool = True,
    ):
        self.keep_mask = keep_mask
        self.n_keep = n_keep
        self.action_vec = action_vec
        self.s_dim = action_logits.size(1)
        self.subaction_mask = subaction_mask

        masks_per_subaction = action_masks.view(
            -1, action_masks.size(-1)
        )  # -> Bool[B*S, A]
        split_masks = torch.split(
            masks_per_subaction, action_vec.tolist(), dim=1
        )  # Tuple[Bool[B*S, A_i], ...]

        logits_per_subaction = action_logits.view(
            -1, action_logits.size(-1)
        )  # Float[B*S, A]
        split_logits = torch.split(
            logits_per_subaction, action_vec.tolist(), dim=1
        )  # Tuple[Float[B*S, A_i], ...]

        self.categoricals = [
            MaskedCategorical(
                logits=lg,
                validate_args=validate_args,
                mask=m,
                entropy_mask_correction=entropy_mask_correction,
            )
            for lg, m in zip(split_logits, split_masks)
        ]

        batch_shape = (
            action_logits.size()[:-1]
            if action_logits.ndimension() > 1
            else torch.Size()
        )
        super().__init__(batch_shape, validate_args=validate_args)

    def log_prob(
        self,
        action: torch.Tensor,  # Int[B, H*W, A]
    ) -> torch.Tensor:
        action_squash = torch.zeros(
            action.size(0),
            self.s_dim,
            action.size(2),
            dtype=action.dtype,
            device=action.device,
        )  # Int[B, S, A]
        for i in range(action.size(0)):
            action_squash[i, : self.n_keep[i], :] = action[i, self.keep_mask[i], :]
        action_per_subaction = action_squash.view(
            -1, action_squash.size(-1)
        ).T  # -> Int[B*S, A] -> Int[A, B*S]

        logprob_per_subaction = []  # List[Float[B*S]], len=A
        for idx, (a, c) in enumerate(
            zip(
                action_per_subaction,
                self.categoricals,
            )
        ):
            if self.subaction_mask and idx in self.subaction_mask:
                reference_index, value = self.subaction_mask[idx]
                logprob_per_subaction.append(
                    torch.where(
                        action_per_subaction[reference_index] == value,
                        c.log_prob(a),
                        0,
                    )
                )
            else:
                logprob_per_subaction.append(c.log_prob(a))
        logprob_stack_per_subaction = torch.stack(
            logprob_per_subaction, dim=-1
        )  # Float[B*S, A]
        if DEBUG_VERIFY:
            non_one_probs = logprob_stack_per_subaction[logprob_stack_per_subaction < 0]
            low_probs = non_one_probs[non_one_probs < -6.90776]
            if len(low_probs):
                logging.warn(
                    f"Found sub 0.1% events: {low_probs.detach().cpu().numpy()}"
                )
            high_probs = non_one_probs[non_one_probs > -0.0010005]
            if len(high_probs):
                logging.warn(
                    f"Found over 99.9% events: {high_probs.detach().cpu().numpy()}"
                )
        logprob_squash = logprob_stack_per_subaction.view(
            -1, action_squash.size(1), len(self.action_vec)
        ).sum(
            dim=(1, 2)
        )  # -> Float[B, S, A] -> Float[B]
        return logprob_squash

    def entropy(self) -> torch.Tensor:
        ent_per_subaction = torch.stack(
            [c.entropy() for c in self.categoricals], dim=-1
        )  # Float[B*S, A]

        return ent_per_subaction.view(-1, self.s_dim, len(self.action_vec)).sum(
            dim=(1, 2)
        )  # -> [B, S, A] -> [B]

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        sample_per_subaction = torch.stack(
            [c.sample(sample_shape) for c in self.categoricals], dim=-1
        )  # Int[B*S, A]
        sample_sequence = sample_per_subaction.view(
            -1, self.s_dim, len(self.action_vec)
        )  # -> Int[B, S, A]

        map_sample = torch.zeros(
            self.keep_mask.size(0),
            self.keep_mask.size(1),
            sample_sequence.size(-1),
            dtype=sample_sequence.dtype,
            device=sample_sequence.device,
        )  # Int[B, H*W, A]
        for i in range(sample_sequence.size(0)):
            map_sample[i, self.keep_mask[i], :] = sample_sequence[
                i, : self.n_keep[i], :
            ]
        return map_sample

    @property
    def mode(self) -> torch.Tensor:
        mode_per_subaction = torch.stack(
            [c.mode for c in self.categoricals], dim=-1
        )  # Int[B*S, A]
        mode_sequence = mode_per_subaction.view(
            -1, self.s_dim, len(self.action_vec)
        )  # -> Int[B, S, A]

        map_mode = torch.zeros(
            self.keep_mask.size(0),
            self.keep_mask.size(1),
            mode_sequence.size(-1),
            dtype=mode_sequence.dtype,
            device=mode_sequence.device,
        )  # Int[B, H*W, A]
        for i in range(mode_sequence.size(0)):
            map_mode[i, self.keep_mask[i], :] = mode_sequence[i, : self.n_keep[i], :]
        return map_mode

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return {}
