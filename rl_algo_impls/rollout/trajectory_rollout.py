from collections import defaultdict
from typing import DefaultDict, Dict, Iterator, List, Optional, TypeVar

import numpy as np
import torch
from gymnasium.spaces import MultiDiscrete
from numpy.typing import NDArray

from rl_algo_impls.rollout.rollout import Batch, BatchMapFn, Rollout
from rl_algo_impls.rollout.rollout import num_actions as get_num_actions
from rl_algo_impls.rollout.trajectory import Trajectory
from rl_algo_impls.shared.actor.gridnet import ValueDependentMask


class TrajectoryRollout(Rollout):
    def __init__(
        self,
        device: torch.device,
        trajectories: List[Trajectory],
        scale_advantage_by_values_accuracy: bool = False,
        full_batch_off_accelerator: bool = True,  # Unused: Assumed True
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
        action_plane_space: Optional[MultiDiscrete] = None,
    ) -> None:
        super().__init__()
        self.device = device
        obs = np.concatenate([t.obs for t in trajectories])
        self.values: NDArray[np.float32] = np.concatenate(
            [t.values for t in trajectories]
        )
        logprobs = np.concatenate([t.logprobs for t in trajectories])
        action_masks = concatenate_actions([t.action_masks for t in trajectories])
        actions = concatenate_actions([t.actions for t in trajectories])

        advantages: NDArray[np.float32] = np.concatenate(
            [t.advantages for t in trajectories]
        )
        num_actions = get_num_actions(
            actions,
            action_masks,
            ValueDependentMask.from_reference_index_to_index_to_value(subaction_mask)
            if subaction_mask
            else None,
            action_plane_space,
        )

        self.returns = advantages + self.values

        assert (
            not scale_advantage_by_values_accuracy
        ), f"{self.__class__.__name__} doesn't implement scale_advantage_by_values_accuracy"

        batch_device = torch.device("cpu")
        self.batch = Batch(
            obs=torch.from_numpy(obs).to(batch_device),
            logprobs=torch.from_numpy(logprobs).to(batch_device)
            if logprobs is not None
            else None,
            actions=actions_to_tensor(actions, batch_device),
            action_masks=actions_to_tensor(action_masks, batch_device),
            num_actions=torch.from_numpy(num_actions).to(batch_device),
            values=torch.from_numpy(self.values).to(batch_device),
            advantages=torch.from_numpy(advantages).to(batch_device),
            returns=torch.from_numpy(self.returns).to(batch_device),
        )

    @property
    def y_true(self) -> np.ndarray:
        return self.returns

    @property
    def y_pred(self) -> np.ndarray:
        return self.values

    @property
    def total_steps(self) -> int:
        return len(self.batch)

    def num_minibatches(self, batch_size: int) -> int:
        return self.total_steps // batch_size

    def add_to_batch(self, map_fn: BatchMapFn, batch_size: int) -> None:
        to_add: DefaultDict[str, List[torch.Tensor]] = defaultdict(list)
        for i in range(0, self.total_steps, batch_size):
            mb_dict = map_fn(
                self.batch[torch.arange(i, i + batch_size)].to(self.device)
            )
            for k, v in mb_dict.items():
                to_add[k].append(v)
        self.batch.additional.update(
            {k: torch.cat(v).to(self.batch.device) for k, v in to_add.items()}
        )

    def minibatches(self, batch_size: int, shuffle: bool = True) -> Iterator[Batch]:
        b_idxs = (
            torch.randperm(self.total_steps)
            if shuffle
            else torch.arange(self.total_steps)
        )
        for i in range(self.num_minibatches(batch_size)):
            mb_idxs = b_idxs[i * batch_size : (i + 1) * batch_size]
            yield self.batch[mb_idxs].to(self.device)


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
