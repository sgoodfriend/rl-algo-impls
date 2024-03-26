from typing import Dict, Iterator, List, NamedTuple, Optional, TypeVar, Union

import numpy as np
import torch
from gymnasium.spaces import MultiDiscrete

from rl_algo_impls.loss.teacher_kl_loss import teacher_kl_loss_enabled
from rl_algo_impls.rollout.rollout import Rollout, flatten_batch_step, num_actions
from rl_algo_impls.rollout.rollout_dataloader import RolloutDataset
from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.actor.gridnet import ValueDependentMask
from rl_algo_impls.shared.data_store.data_store_data import RolloutView
from rl_algo_impls.shared.gae import compute_advantages
from rl_algo_impls.shared.tensor_utils import (
    TDN,
    NumOrArray,
    NumpyOrDict,
    TensorOrDict,
    get_items,
    numpy_to_tensor,
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

    num_actions: Optional[torch.Tensor] = None

    def to(self: PPOBatchSelf, device: torch.device) -> PPOBatchSelf:
        def to_device(t: TDN) -> TDN:
            if t is None:
                return t
            elif isinstance(t, dict):
                return {k: v.to(device) for k, v in t.items()}  # type: ignore
            else:
                return t.to(device)

        return self.__class__(*(to_device(t) for t in self))

    def __getitem__(
        self: PPOBatchSelf, indices: Union[int, List[int], torch.Tensor]
    ) -> PPOBatchSelf:
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
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
        action_plane_space: Optional[MultiDiscrete] = None,
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

        if config.algo_hyperparams.get(
            "scale_loss_by_has_actions", False
        ) or config.algo_hyperparams.get("scale_loss_by_num_actions", False):
            n_actions = num_actions(
                actions,
                action_masks,
                (
                    ValueDependentMask.from_reference_index_to_index_to_value(
                        subaction_mask
                    )
                    if subaction_mask
                    else None
                ),
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

    def batch(self, device: torch.device) -> PPOBatch:
        if self._batch is not None:
            return self._batch.to(device)
        self._batch = PPOBatch(
            numpy_to_tensor(self.obs, device),
            numpy_to_tensor(self.actions, device),
            (
                numpy_to_tensor(self.action_masks, device)
                if self.action_masks is not None
                else None
            ),
            numpy_to_tensor(self.logprobs, device),
            numpy_to_tensor(self.values, device),
            numpy_to_tensor(self.advantages, device),
            numpy_to_tensor(self.returns, device),
            (
                numpy_to_tensor(self.teacher_logprobs, device)
                if self.teacher_logprobs is not None
                else None
            ),
            (
                numpy_to_tensor(self.num_actions, device)
                if self.num_actions is not None
                else None
            ),
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

    def dataset(self, device: torch.device) -> RolloutDataset:
        device = torch.device("cpu") if self.full_batch_off_accelerator else device
        return RolloutDataset(self.batch(device))
