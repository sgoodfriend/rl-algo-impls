from typing import Iterator, NamedTuple, Optional, TypeVar

import numpy as np
import torch

from rl_algo_impls.loss.teacher_kl_loss import teacher_kl_loss_enabled
from rl_algo_impls.rollout.rollout import Rollout, flatten_batch_step
from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.data_store.data_store_data import RolloutView
from rl_algo_impls.shared.gae import compute_advantages
from rl_algo_impls.shared.tensor_utils import (
    TDN,
    NumOrArray,
    NumpyOrDict,
    TensorOrDict,
    get_items,
    numpy_to_tensor,
    to_device,
)

PPOBatchSelf = TypeVar("PPOBatchSelf", bound="PPOBatch")


class PPOBatch(NamedTuple):
    obs: torch.Tensor

    actions: TensorOrDict
    action_masks: Optional[TensorOrDict]

    logprobs: torch.Tensor

    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    teacher_logprobs: Optional[torch.Tensor]

    def to(self: PPOBatchSelf, device: torch.device) -> PPOBatchSelf:
        return self.__class__(*(to_device(t, device) for t in self))

    def __getitem__(self: PPOBatchSelf, indices: torch.Tensor) -> PPOBatchSelf:
        def by_indices_fn(_t: TDN) -> TDN:
            if _t is None:
                return _t
            if isinstance(_t, dict):
                return {k: v[indices] for k, v in _t.items()}
            return _t[indices]

        return self.__class__(*(by_indices_fn(t) for t in self))


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
        self.obs = numpy_to_tensor(flatten_batch_step(obs), cpu_device)
        self.actions = numpy_to_tensor(flatten_batch_step(actions), cpu_device)
        self.values = numpy_to_tensor(flatten_batch_step(values), cpu_device)
        self.logprobs = numpy_to_tensor(logprobs.reshape(-1), cpu_device)
        self.action_masks = (
            numpy_to_tensor(flatten_batch_step(action_masks), cpu_device)
            if action_masks is not None
            else None
        )
        self.full_batch_off_accelerator = full_batch_off_accelerator

        self.advantages = numpy_to_tensor(
            flatten_batch_step(
                compute_advantages(
                    rewards,
                    values,
                    episode_starts,
                    next_episode_starts,
                    next_values,
                    gamma,
                    gae_lambda,
                )
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
            self.teacher_logprobs = numpy_to_tensor(
                teacher_logprobs.reshape(-1), cpu_device
            )
        else:
            self.teacher_logprobs = None

    @property
    def y_true(self) -> np.ndarray:
        return self.returns.numpy()

    @property
    def y_pred(self) -> np.ndarray:
        return self.values.numpy()

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
            *(
                to_device(t, device)
                for t in (
                    self.obs,
                    self.actions,
                    self.action_masks,
                    self.logprobs,
                    self.values,
                    self.advantages,
                    self.returns,
                    self.teacher_logprobs,
                )
            )
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
