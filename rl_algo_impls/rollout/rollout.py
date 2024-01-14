import dataclasses
from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass
from typing import Callable, Dict, Iterator, Optional, TypeVar

import numpy as np
import torch
from gymnasium.spaces import MultiDiscrete

from rl_algo_impls.shared.actor.gridnet import ValueDependentMask
from rl_algo_impls.shared.tensor_utils import NumpyOrDict, TensorOrDict

BatchSelf = TypeVar("BatchSelf", bound="Batch")
TDN = TypeVar("TDN", torch.Tensor, Dict[str, torch.Tensor], None)


@dataclass
class Batch:
    obs: torch.Tensor
    logprobs: Optional[torch.Tensor]

    actions: TensorOrDict
    action_masks: Optional[TensorOrDict]
    num_actions: Optional[torch.Tensor]

    values: torch.Tensor

    advantages: torch.Tensor
    returns: torch.Tensor
    additional: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)

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

    def __getitem__(self: BatchSelf, indices: torch.Tensor) -> BatchSelf:
        def by_indices_fn(_t: TDN) -> TDN:
            if _t is None:
                return _t
            if isinstance(_t, dict):
                return {k: v[indices] for k, v in _t.items()}
            return _t[indices]

        return self.__class__(
            by_indices_fn(self.obs),
            by_indices_fn(self.logprobs),
            by_indices_fn(self.actions),
            by_indices_fn(self.action_masks),
            by_indices_fn(self.num_actions),
            by_indices_fn(self.values),
            by_indices_fn(self.advantages),
            by_indices_fn(self.returns),
            by_indices_fn(self.additional),
        )

    def __len__(self) -> int:
        return self.obs.shape[0]


BatchMapFn = Callable[[Batch], Dict[str, torch.Tensor]]


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
    ) -> Iterator[Batch]:
        ...

    def add_to_batch(
        self, map_fn: BatchMapFn, batch_size: int, device: torch.device
    ) -> None:
        ...


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
