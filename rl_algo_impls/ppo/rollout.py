from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass
from typing import Iterator, Optional, Union

import numpy as np
import torch

from rl_algo_impls.shared.gae import compute_advantages
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.tensor_utils import (
    NumpyOrDict,
    TensorOrDict,
    tensor_by_indicies,
)


@dataclass
class Batch:
    obs: torch.Tensor
    logprobs: torch.Tensor

    actions: TensorOrDict
    action_masks: Optional[TensorOrDict]

    values: torch.Tensor

    advantages: torch.Tensor
    returns: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.obs.device


class Rollout:
    next_obs: np.ndarray
    next_episode_starts: np.ndarray

    obs: np.ndarray
    actions: NumpyOrDict
    rewards: np.ndarray
    episode_starts: np.ndarray
    values: np.ndarray
    logprobs: np.ndarray
    action_masks: Optional[NumpyOrDict]

    advantages: Optional[np.ndarray] = None
    returns: Optional[np.ndarray] = None

    _y_true: Optional[np.ndarray] = None

    _batch: Optional[Batch] = None

    def __init__(
        self,
        next_obs: np.ndarray,
        next_episode_starts: np.ndarray,
        obs: np.ndarray,
        actions: NumpyOrDict,
        rewards: np.ndarray,
        episode_starts: np.ndarray,
        values: np.ndarray,
        logprobs: np.ndarray,
        action_masks: Optional[NumpyOrDict],
        scale_advantage_by_values_accuracy: bool = False,
    ) -> None:
        self.next_obs = next_obs
        self.next_episode_starts = next_episode_starts

        self.obs = obs
        self.actions = actions
        self.rewards = rewards
        self.episode_starts = episode_starts
        self.values = values
        self.logprobs = logprobs
        self.action_masks = action_masks
        self.scale_advantage_by_values_accuracy = scale_advantage_by_values_accuracy

    def update_advantages(
        self,
        policy: ActorCritic,
        gamma: Union[float, np.ndarray],
        gae_lambda: Union[float, np.ndarray],
        update_returns: bool = False,
    ):
        self.advantages = compute_advantages(
            self.rewards,
            self.values,
            self.episode_starts,
            self.next_episode_starts,
            self.next_obs,
            policy,
            gamma,
            gae_lambda,
        )

        if self.returns is None or update_returns:
            self.returns = self.advantages + self.values
            self._y_true = self.returns.reshape((-1,) + self.returns.shape[2:])
            if self._batch:
                self._batch.returns = torch.tensor(self._y_true).to(self._batch.device)

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
    def y_true(self) -> np.ndarray:
        assert self._y_true is not None, "Must run update_advantages before y_true"
        return self._y_true

    @property
    def total_steps(self) -> int:
        return int(np.prod(self.rewards.shape[:2]))

    def num_minibatches(self, batch_size: int) -> int:
        return self.total_steps // batch_size + (
            1 if self.total_steps % batch_size else 0
        )

    def minibatches(self, batch_size: int, device: torch.device) -> Iterator[Batch]:
        if self._batch is None:
            b_obs = flatten_to_tensor(self.obs, device)
            b_logprobs = torch.tensor(self.logprobs.reshape(-1)).to(device)

            b_actions = flatten_actions_to_tensor(self.actions, device)
            b_action_masks = (
                flatten_actions_to_tensor(self.action_masks, device)
                if self.action_masks is not None
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
                b_values,
                b_advantages,
                b_returns,
            )

        obs, logprobs, actions, action_masks, values, advantages, returns = astuple(
            self._batch
        )
        b_idxs = torch.randperm(self.total_steps)
        for i in range(0, self.total_steps, batch_size):
            mb_idxs = b_idxs[i : i + batch_size]
            yield Batch(
                obs[mb_idxs],
                logprobs[mb_idxs],
                tensor_by_indicies(actions, mb_idxs),
                tensor_by_indicies(action_masks, mb_idxs)
                if action_masks is not None
                else None,
                values[mb_idxs],
                advantages[mb_idxs],
                returns[mb_idxs],
            )


class RolloutGenerator(ABC):
    def __init__(
        self,
        n_steps: int,
        sde_sample_freq: int,
        scale_advantage_by_values_accuracy: bool = False,
    ) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.sde_sample_freq = sde_sample_freq
        self.scale_advantage_by_values_accuracy = scale_advantage_by_values_accuracy

    @abstractmethod
    def rollout(self) -> Rollout:
        ...


def flatten_to_tensor(a: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(a.reshape((-1,) + a.shape[2:])).to(device)


def flatten_actions_to_tensor(a: NumpyOrDict, device: torch.device) -> TensorOrDict:
    if isinstance(a, dict):
        return {k: flatten_to_tensor(v, device) for k, v in a.items()}
    return flatten_to_tensor(a, device)
