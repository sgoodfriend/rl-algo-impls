from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch

from rl_algo_impls.loss.teacher_kl_loss import teacher_kl_loss_enabled
from rl_algo_impls.rollout.rollout import (
    Batch,
    Rollout,
    flatten_actions_to_tensor,
    flatten_to_tensor,
)
from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.data_store.data_store_data import RolloutView
from rl_algo_impls.shared.gae import compute_advantages
from rl_algo_impls.shared.tensor_utils import NumOrArray, NumpyOrDict, get_items


@dataclass
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
        cpu_device = torch.device("cpu")
        self.obs = flatten_to_tensor(obs, cpu_device)
        self.actions = flatten_actions_to_tensor(actions, cpu_device)
        self.values = flatten_to_tensor(values, cpu_device)
        self.logprobs = torch.tensor(logprobs.reshape(-1)).to(cpu_device)
        self.action_masks = (
            flatten_actions_to_tensor(action_masks, cpu_device)
            if action_masks is not None
            else None
        )
        self.full_batch_off_accelerator = full_batch_off_accelerator

        self.advantages = flatten_to_tensor(
            compute_advantages(
                rewards,
                values,
                episode_starts,
                next_episode_starts,
                next_values,
                gamma,
                gae_lambda,
            ),
            cpu_device,
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
            self.teacher_logprobs = torch.tensor(teacher_logprobs.reshape(-1)).to(
                cpu_device
            )
        else:
            self.teacher_logprobs = None

    @property
    def y_true(self) -> torch.Tensor:
        return self.returns

    @property
    def y_pred(self) -> torch.Tensor:
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
            self.obs,
            self.actions,
            self.action_masks,
            self.logprobs,
            self.values,
            self.advantages,
            self.returns,
            self.teacher_logprobs,
        ).to(device)
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
