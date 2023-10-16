from typing import Dict, Iterator, Optional

import numpy as np
import torch
from gymnasium.spaces import MultiDiscrete

from rl_algo_impls.rollout.rollout import (
    Batch,
    Rollout,
    flatten_actions_to_tensor,
    flatten_to_tensor,
    num_actions,
)
from rl_algo_impls.shared.actor.gridnet import ValueDependentMask
from rl_algo_impls.shared.gae import compute_advantages
from rl_algo_impls.shared.tensor_utils import NumOrArray, NumpyOrDict


class VecRollout(Rollout):
    obs: np.ndarray
    actions: NumpyOrDict
    rewards: np.ndarray
    episode_starts: np.ndarray
    values: np.ndarray
    logprobs: Optional[np.ndarray]
    action_masks: Optional[NumpyOrDict]
    num_actions: Optional[np.ndarray]

    advantages: np.ndarray
    returns: np.ndarray

    _y_true: np.ndarray

    _batch: Optional[Batch] = None

    def __init__(
        self,
        next_episode_starts: np.ndarray,
        next_values: np.ndarray,
        obs: np.ndarray,
        actions: NumpyOrDict,
        rewards: np.ndarray,
        episode_starts: np.ndarray,
        values: np.ndarray,
        logprobs: Optional[np.ndarray],
        action_masks: Optional[NumpyOrDict],
        gamma: NumOrArray,
        gae_lambda: NumOrArray,
        scale_advantage_by_values_accuracy: bool = False,
        full_batch_off_accelerator: bool = False,
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
        action_plane_space: Optional[MultiDiscrete] = None,
    ) -> None:
        super().__init__()
        self.obs = obs
        self.actions = actions
        self.rewards = rewards
        self.episode_starts = episode_starts
        self.values = values
        self.logprobs = logprobs
        self.action_masks = action_masks
        self.scale_advantage_by_values_accuracy = scale_advantage_by_values_accuracy
        self.full_batch_off_accelerator = full_batch_off_accelerator

        self.num_actions = num_actions(
            actions,
            action_masks,
            ValueDependentMask.from_reference_index_to_index_to_value(subaction_mask)
            if subaction_mask
            else None,
            action_plane_space,
        )

        self.advantages = compute_advantages(
            self.rewards,
            self.values,
            self.episode_starts,
            next_episode_starts,
            next_values,
            gamma,
            gae_lambda,
        )

        self.returns = self.advantages + self.values
        self._y_true = self.returns.reshape((-1,) + self.returns.shape[2:])

        if self.scale_advantage_by_values_accuracy:
            self.advantages *= np.exp(
                -np.abs(self.values - self.returns) / self.returns.ptp()
            )

    @property
    def y_true(self) -> np.ndarray:
        return self._y_true

    @property
    def y_pred(self) -> np.ndarray:
        return self.values.reshape((-1,) + self.values.shape[2:])

    @property
    def total_steps(self) -> int:
        return int(np.prod(self.rewards.shape[:2]))

    def num_minibatches(self, batch_size: int) -> int:
        return self.total_steps // batch_size + (
            1 if self.total_steps % batch_size else 0
        )

    def batch(self, device: torch.device) -> Batch:
        if self._batch is None:
            b_obs = flatten_to_tensor(self.obs, device)
            b_logprobs = (
                torch.tensor(self.logprobs.reshape(-1)).to(device)
                if self.logprobs is not None
                else None
            )

            b_actions = flatten_actions_to_tensor(self.actions, device)
            b_action_masks = (
                flatten_actions_to_tensor(self.action_masks, device)
                if self.action_masks is not None
                else None
            )
            b_num_actions = (
                torch.tensor(self.num_actions.reshape(-1)).to(device)
                if self.num_actions is not None
                else None
            )

            b_values = torch.tensor(self.y_pred).to(device)

            assert (
                self.advantages is not None and self.returns is not None
            ), "Must call update_advantages before minibatches"
            b_advantages = flatten_to_tensor(self.advantages, device)
            b_returns = torch.tensor(self._y_true).to(device)

            self._batch = Batch(
                b_obs,
                b_logprobs,
                b_actions,
                b_action_masks,
                b_num_actions,
                b_values,
                b_advantages,
                b_returns,
            )
        return self._batch.to(device)

    def minibatches(self, batch_size: int, device: torch.device) -> Iterator[Batch]:
        batch = self.batch(
            torch.device("cpu") if self.full_batch_off_accelerator else device
        )
        b_idxs = torch.randperm(self.total_steps)
        for i in range(0, self.total_steps, batch_size):
            mb_idxs = b_idxs[i : i + batch_size]
            yield batch[mb_idxs].to(device)
