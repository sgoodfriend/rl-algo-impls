from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch

from rl_algo_impls.loss.teacher_kl_loss import teacher_kl_loss_enabled
from rl_algo_impls.rollout.rollout import Batch, Rollout, flatten_batch_step
from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.data_store.data_store_data import RolloutView
from rl_algo_impls.shared.gae import compute_advantages
from rl_algo_impls.shared.tensor_utils import (
    NumOrArray,
    NumpyOrDict,
    get_items,
    numpy_to_tensor,
)


class PPOBatch(Batch):
    logprobs: torch.Tensor

    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    teacher_logprobs: Optional[torch.Tensor]


class PPORollout(Rollout):
    _batch: Optional[PPOBatch] = None

    def __init__(
        self,
        config: Config,
        rollout_view: RolloutView,
        next_episode_starts: np.ndarray,
        next_values: np.ndarray,
        obs: np.ndarray,
        actions: NumpyOrDict,
        rewards: np.ndarray,
        episode_starts: np.ndarray,
        values: np.ndarray,
        logprobs: np.ndarray,
        action_masks: Optional[NumpyOrDict],
        gamma: NumOrArray,
        gae_lambda: NumOrArray,
        full_batch_off_accelerator: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.obs = flatten_batch_step(obs)
        self.actions = flatten_batch_step(actions)
        self.values = flatten_batch_step(values)
        self.logprobs = logprobs.reshape(-1)
        self.action_masks = (
            flatten_batch_step(action_masks) if action_masks is not None else None
        )
        self.full_batch_off_accelerator = full_batch_off_accelerator

        self.advantages = flatten_batch_step(
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

        self.returns = self.advantages + self.values

        if teacher_kl_loss_enabled(config) and rollout_view.latest_checkpoint_policy:
            teacher_logprobs = np.zeros_like(logprobs)
            teacher_policy = rollout_view.latest_checkpoint_policy
            for idx, o_batch in enumerate(obs):
                action_batch = get_items(actions, idx)
                action_mask_batch = (
                    get_items(action_masks, idx) if action_masks is not None else None
                )
                teacher_logprobs[idx] = teacher_policy.logprobs(
                    o_batch, action_batch, action_mask_batch
                )
            self.teacher_logprobs = teacher_logprobs.reshape(-1)
        else:
            self.teacher_logprobs = None

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

    def batch(self, device: torch.device) -> PPOBatch:
        if self._batch is not None:
            return self._batch.to(device)
        self._batch = PPOBatch(
            numpy_to_tensor(self.obs, device),
            numpy_to_tensor(self.actions, device),
            numpy_to_tensor(self.action_masks, device)
            if self.action_masks is not None
            else None,
            numpy_to_tensor(self.logprobs, device),
            numpy_to_tensor(self.values, device),
            numpy_to_tensor(self.advantages, device),
            numpy_to_tensor(self.returns, device),
            numpy_to_tensor(self.teacher_logprobs, device)
            if self.teacher_logprobs is not None
            else None,
        )
        return self._batch

    def minibatches(
        self, batch_size: int, device: torch.device, shuffle: bool = True
    ) -> Iterator[PPOBatch]:
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