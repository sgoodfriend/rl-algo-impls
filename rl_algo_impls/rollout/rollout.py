from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, NamedTuple, Optional, Tuple, TypeVar

import numpy as np
import torch
from gymnasium.spaces import MultiDiscrete

from rl_algo_impls.shared.actor.gridnet import ValueDependentMask
from rl_algo_impls.shared.tensor_utils import TDN, NumpyOrDict, TensorOrDict

BatchTuple = Tuple[TDN, ...]


class Rollout(ABC):
    @property
    @abstractmethod
    def y_true(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def y_pred(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def total_steps(self) -> int:
        ...

    @abstractmethod
    def num_minibatches(self, batch_size: int) -> int:
        ...

    @abstractmethod
    def minibatches(
        self, batch_size: int, device: torch.device, shuffle: bool = True
    ) -> Iterator[BatchTuple]:
        ...


ND = TypeVar("ND", np.ndarray, Dict[str, np.ndarray])


def flatten_batch_step(a: ND) -> ND:
    def flatten_array_fn(_a: np.ndarray) -> np.ndarray:
        return _a.reshape((-1,) + _a.shape[2:])

    if isinstance(a, dict):
        return {k: flatten_array_fn(v) for k, v in a.items()}
    return flatten_array_fn(a)


def flatten_to_tensor(a: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(a.reshape((-1,) + a.shape[2:])).to(device)


def flatten_actions_to_tensor(a: NumpyOrDict, device: torch.device) -> TensorOrDict:
    if isinstance(a, dict):
        return {k: flatten_to_tensor(v, device) for k, v in a.items()}
    return flatten_to_tensor(a, device)


def num_actions(
    actions: NumpyOrDict,
    action_masks: Optional[NumpyOrDict],
    subaction_mask: Optional[Dict[int, ValueDependentMask]],
    action_plane_space: Optional[MultiDiscrete],
) -> Optional[np.ndarray]:
    if action_masks is None:
        return None

    if isinstance(action_masks, dict):
        per_position_actions = per_position_num_actions(
            actions["per_position"],
            action_masks["per_position"],
            subaction_mask,
            action_plane_space,
        )
        pick_positions = action_masks["pick_position"].any(axis=-2).sum(axis=-1)
        return (
            per_position_actions
            + np.where(pick_positions > 0, np.log(pick_positions), 0)
        ).astype(np.float32)
    else:
        assert isinstance(actions, np.ndarray)
        return per_position_num_actions(
            actions, action_masks, subaction_mask, action_plane_space
        )


def per_position_num_actions(
    actions: np.ndarray,
    action_masks: np.ndarray,
    subaction_mask: Optional[Dict[int, ValueDependentMask]],
    action_plane_space: Optional[MultiDiscrete],
) -> np.ndarray:
    if not subaction_mask:
        return np.sum(np.any(action_masks, axis=-1), axis=-1)
    assert action_plane_space
    num_actions = np.zeros(actions.shape[:-2], dtype=np.int32)
    m_idx = 0
    for idx, m_sz in enumerate(action_plane_space.nvec):
        m = action_masks[..., m_idx : m_idx + m_sz]
        if idx in subaction_mask:
            reference_index, value = subaction_mask[idx]
            m = np.where(
                np.expand_dims(actions[..., reference_index] == value, axis=-1),
                m,
                False,
            )
        num_actions += np.sum(np.any(m, axis=-1), axis=-1)
        m_idx += m_sz
    return num_actions
