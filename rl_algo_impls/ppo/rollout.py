from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass
from typing import Dict, Iterator, Optional, TypeVar, Union

import numpy as np
import torch

from rl_algo_impls.shared.gae import compute_advantages
from rl_algo_impls.shared.tensor_utils import (
    NumOrArray,
    NumpyOrDict,
    TensorOrDict,
    tensor_by_indicies,
)

BatchSelf = TypeVar("BatchSelf", bound="Batch")
TDN = TypeVar("TDN", torch.Tensor, Dict[str, torch.Tensor], None)


@dataclass
class Batch:
    obs: torch.Tensor
    logprobs: Optional[torch.Tensor]

    actions: TensorOrDict
    action_masks: Optional[TensorOrDict]
    num_actions: Optional[TensorOrDict]

    values: torch.Tensor

    advantages: torch.Tensor
    returns: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.obs.device

    def to(self: BatchSelf, device: torch.device) -> BatchSelf:
        if self.device == device:
            return self

        def to_device(t: TDN) -> TDN:
            if t is None:
                return t
            elif isinstance(t, dict):
                return {k: v.to(device) for k, v in t.items()}  # type: ignore
            else:
                return t.to(device)

        return self.__class__(*(to_device(t) for t in astuple(self)))


class Rollout:
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

    y_true: np.ndarray

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
    ) -> None:
        self.obs = obs
        self.actions = actions
        self.rewards = rewards
        self.episode_starts = episode_starts
        self.values = values
        self.logprobs = logprobs
        self.action_masks = action_masks
        self.scale_advantage_by_values_accuracy = scale_advantage_by_values_accuracy
        self.full_batch_off_accelerator = full_batch_off_accelerator

        if self.action_masks is not None:
            if isinstance(self.action_masks, dict):
                self.num_actions = np.sum(
                    [
                        np.sum(np.any(am, axis=-1), axis=-1)
                        for am in self.action_masks.values()
                    ],
                    axis=0,
                )
            else:
                self.num_actions = np.sum(np.any(self.action_masks, axis=-1), axis=-1)
        else:
            self.num_actions = None

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
        self.y_true = self.returns.reshape((-1,) + self.returns.shape[2:])
        if self._batch:
            self._batch.returns = torch.tensor(self.y_true).to(self._batch.device)

        if self.scale_advantage_by_values_accuracy:
            self.advantages *= np.exp(
                -np.abs(self.values - self.returns) / self.returns.ptp()
            )
        if self._batch:
            self._batch.advantages = flatten_to_tensor(
                self.advantages, self._batch.device
            )

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
            b_returns = torch.tensor(self.y_true).to(device)

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
        (
            obs,
            logprobs,
            actions,
            action_masks,
            num_actions,
            values,
            advantages,
            returns,
        ) = astuple(
            self.batch(
                torch.device("cpu") if self.full_batch_off_accelerator else device
            )
        )
        b_idxs = torch.randperm(self.total_steps)
        for i in range(0, self.total_steps, batch_size):
            mb_idxs = b_idxs[i : i + batch_size]
            yield Batch(
                obs[mb_idxs],
                logprobs[mb_idxs] if logprobs is not None else None,
                tensor_by_indicies(actions, mb_idxs),
                tensor_by_indicies(action_masks, mb_idxs)
                if action_masks is not None
                else None,
                num_actions[mb_idxs] if num_actions is not None else None,
                values[mb_idxs],
                advantages[mb_idxs],
                returns[mb_idxs],
            ).to(device)


class RolloutGenerator(ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def rollout(self, gamma: NumOrArray, gae_lambda: NumOrArray) -> Rollout:
        ...


def flatten_to_tensor(a: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(a.reshape((-1,) + a.shape[2:])).to(device)


def flatten_actions_to_tensor(a: NumpyOrDict, device: torch.device) -> TensorOrDict:
    if isinstance(a, dict):
        return {k: flatten_to_tensor(v, device) for k, v in a.items()}
    return flatten_to_tensor(a, device)
