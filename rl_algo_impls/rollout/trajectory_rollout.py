from typing import Dict, Iterator, List, Optional, TypeVar

import numpy as np
import torch
from gym.spaces import MultiDiscrete
from numpy.typing import NDArray

from rl_algo_impls.rollout.rollout import Batch, Rollout, num_actions
from rl_algo_impls.rollout.trajectory import Trajectory
from rl_algo_impls.shared.actor.gridnet import ValueDependentMask


class TrajectoryRollout(Rollout):
    def __init__(
        self,
        trajectories: List[Trajectory],
        scale_advantage_by_values_accuracy: bool = False,
        full_batch_off_accelerator: bool = False,
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
        action_plane_space: Optional[MultiDiscrete] = None,
    ) -> None:
        self.full_batch_off_accelerator = full_batch_off_accelerator

        self.obs = np.concatenate([t.obs for t in trajectories])
        self.rewards = np.concatenate([t.rewards for t in trajectories])
        self.values: NDArray[np.float32] = np.concatenate(
            [t.values for t in trajectories]
        )
        self.logprobs = np.concatenate([t.logprobs for t in trajectories])
        self.action_masks = concatenate_actions([t.action_masks for t in trajectories])
        self.actions = concatenate_actions([t.actions for t in trajectories])

        self.advantages: NDArray[np.float32] = np.concatenate(
            [t.advantages for t in trajectories]
        )
        self.num_actions = num_actions(
            self.actions,
            self.action_masks,
            ValueDependentMask.from_reference_index_to_index_to_value(subaction_mask)
            if subaction_mask
            else None,
            action_plane_space,
        )

        self.returns = self.advantages + self.values

        assert (
            not scale_advantage_by_values_accuracy
        ), f"{self.__class__.__name__} doesn't implement scale_advantage_by_values_accuracy"

    @property
    def y_true(self) -> np.ndarray:
        return self.returns

    @property
    def y_pred(self) -> np.ndarray:
        return self.values

    @property
    def total_steps(self) -> int:
        return len(self.obs)

    def num_minibatches(self, batch_size: int) -> int:
        return self.total_steps // batch_size

    def minibatches(self, batch_size: int, device: torch.device) -> Iterator[Batch]:
        batch_device = (
            torch.device("cpu") if self.full_batch_off_accelerator else device
        )
        batch = Batch(
            obs=torch.from_numpy(self.obs).to(batch_device),
            logprobs=torch.from_numpy(self.logprobs).to(batch_device)
            if self.logprobs is not None
            else None,
            actions=actions_to_tensor(self.actions, batch_device),
            action_masks=actions_to_tensor(self.action_masks, batch_device),
            num_actions=torch.from_numpy(self.num_actions).to(batch_device),
            values=torch.from_numpy(self.values).to(batch_device),
            advantages=torch.from_numpy(self.advantages).to(batch_device),
            returns=torch.from_numpy(self.returns).to(batch_device),
        )
        b_idxs = torch.randperm(self.total_steps)
        for i in range(self.num_minibatches(batch_size)):
            mb_idxs = b_idxs[i * batch_size : (i + 1) * batch_size]
            yield batch[mb_idxs].to(device)


ND = TypeVar("ND", NDArray, Dict[str, NDArray], None)


def concatenate_actions(actions: List[ND]) -> ND:
    if isinstance(actions[0], dict):
        return {k: np.concatenate([a[k] for a in actions]) for k in actions[0]}
    elif actions[0] is None:
        return None
    return np.concatenate(actions)


def actions_to_tensor(actions: ND, device: torch.device) -> ND:
    if isinstance(actions, dict):
        return {k: torch.from_numpy(v).to(device) for k, v in actions.items()}
    elif actions is None:
        return None
    return torch.from_numpy(actions).to(device)
