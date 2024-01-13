from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import numpy as np
import torch
from gymnasium.spaces import MultiDiscrete

from rl_algo_impls.rollout.rollout import (
    Batch,
    Rollout,
    flatten_batch_step,
    num_actions,
)
from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.actor.gridnet import ValueDependentMask
from rl_algo_impls.shared.gae import compute_advantages
from rl_algo_impls.shared.tensor_utils import NumOrArray, NumpyOrDict


@dataclass
class ACBCBatch(Batch):
    returns: torch.Tensor

    num_actions: Optional[torch.Tensor] = None


class ACBCRollout(Rollout):
    def __init__(
        self,
        config: Config,
        next_episode_starts: np.ndarray,
        next_values: np.ndarray,
        obs: np.ndarray,
        actions: NumpyOrDict,
        rewards: np.ndarray,
        episode_starts: np.ndarray,
        values: np.ndarray,
        action_masks: Optional[NumpyOrDict],
        gamma: NumOrArray,
        gae_lambda: NumOrArray,
        full_batch_off_accelerator: bool = False,
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
        action_plane_space: Optional[MultiDiscrete] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.obs = flatten_batch_step(obs)
        self.actions = flatten_batch_step(actions)
        self.values = flatten_batch_step(values)
        self.action_masks = (
            flatten_batch_step(action_masks) if action_masks is not None else None
        )
        self.full_batch_off_accelerator = full_batch_off_accelerator

        advantages = flatten_batch_step(
            compute_advantages(
                rewards,
                values,
                episode_starts,
                next_episode_starts,
                next_values,
                gamma,
                gae_lambda,
            )
        )

        self.returns = advantages + self.values

        if config.algo_hyperparams.get("scale_loss_by_num_actions", False):
            n_actions = num_actions(
                actions,
                action_masks,
                ValueDependentMask.from_reference_index_to_index_to_value(
                    subaction_mask
                )
                if subaction_mask
                else None,
                action_plane_space,
            )
            self.num_actions = n_actions.reshape(-1) if n_actions is not None else None
        else:
            self.num_actions = None

    @property
    def y_true(self) -> np.ndarray:
        return self.returns

    @property
    def y_pred(self) -> np.ndarray:
        return self.values

    @property
    def total_steps(self) -> int:
        return self.obs.shape[0]

    def num_minibatches(self, batch_size: int) -> int:
        return self.total_steps // batch_size + (
            1 if self.total_steps % batch_size else 0
        )

    def batch(self, device: torch.device) -> ACBCBatch:
        if self._batch is not None:
            return self._batch.to(device)
        self._batch = ACBCBatch(
            numpy_to_tensor(self.obs, device),
            numpy_to_tensor(self.actions, device),
            numpy_to_tensor(self.action_masks, device)
            if self.action_masks is not None
            else None,
            numpy_to_tensor(self.returns, device),
            numpy_to_tensor(self.num_actions, device)
            if self.num_actions is not None
            else None,
        )
        return self._batch

    def minibatches(
        self, batch_size: int, device: torch.device, shuffle: bool = True
    ) -> Iterator[ACBCBatch]:
        batch = self.batch(
            torch.device("cpu") if self.full_batch_off_accelerator else device
        )
        b_idxs = (
            torch.randperm(self.total_steps)
            if shuffle
            else torch.arange(self.total_steps)
        )
        for i in range(0, self.total_steps, batch_size):
            mb_idxs = b_idxs[i : i + batch_size]
            yield batch[mb_idxs].to(device)
