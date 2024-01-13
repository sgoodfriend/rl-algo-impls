from typing import Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch


def expand_dims_to_match(a: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    assert (
        a.shape == shape[: len(a.shape)]
    ), f"Array {a.shape} must match early dims of {shape}"
    while len(a.shape) < len(shape):
        a = np.expand_dims(a, -1)
    return a


def unqueeze_dims_to_match(t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    assert (
        t.shape == shape[: len(t.shape)]
    ), f"Tensor {t.shape} must match early dims of {shape}"
    while len(t.shape) < len(shape):
        t = t.unsqueeze(-1)
    return t


def prepend_dims_to_match(a: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    assert (
        a.shape == shape[-len(a.shape) :]
    ), f"Array {a.shape} must match later dims of {shape}"
    while len(a.shape) < len(shape):
        a = np.expand_dims(a, 0)
    return a


NumOrList = Union[float, int, List[float], List[int]]
NumOrArray = Union[float, np.ndarray]


def num_or_array(num_or_list: NumOrList) -> NumOrArray:
    if isinstance(num_or_list, list):
        return np.array(num_or_list)
    return num_or_list


TensorOrDict = Union[torch.Tensor, Dict[str, torch.Tensor]]
NumpyOrDict = Union[np.ndarray, Dict[str, np.ndarray]]


def tensor_to_numpy(t: TensorOrDict) -> NumpyOrDict:
    def to_numpy_fn(_tensor: torch.Tensor) -> np.ndarray:
        return _tensor.cpu().numpy()

    if isinstance(t, dict):
        return {k: to_numpy_fn(v) for k, v in t.items()}
    return to_numpy_fn(t)


def numpy_to_tensor(a: NumpyOrDict, device: torch.device) -> TensorOrDict:
    def to_tensor_fn(_a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(_a).to(device)

    if isinstance(a, dict):
        return {k: to_tensor_fn(v) for k, v in a.items()}
    return to_tensor_fn(a)


def tensor_by_indices(t: TensorOrDict, idxs: torch.Tensor) -> TensorOrDict:
    def by_indices_fn(_t: torch.Tensor) -> torch.Tensor:
        return _t[idxs]

    if isinstance(t, dict):
        return {k: by_indices_fn(v) for k, v in t.items()}
    return by_indices_fn(t)


def batch_dict_keys(a: Optional[np.ndarray]) -> Optional[NumpyOrDict]:
    if a is None:
        return None
    if a.dtype.char == "O":
        return {k: np.array([a_i[k] for a_i in a]) for k in a[0]}
    return a


ND = TypeVar("ND", np.ndarray, Dict[str, np.ndarray])


def set_items(destination: ND, subset: ND, idx: Union[int, np.ndarray]):
    def fn(_d: np.ndarray, _s: np.ndarray):
        _d[idx] = _s

    if isinstance(destination, dict):
        assert isinstance(subset, dict)
        for k, d in destination.items():
            fn(d, subset[k])
    else:
        assert isinstance(subset, np.ndarray)
        fn(destination, subset)


def get_items(source: ND, idx: Union[int, np.ndarray]) -> ND:
    def fn(_s: np.ndarray) -> np.ndarray:
        return _s[idx]

    if isinstance(source, dict):
        return {k: fn(v) for k, v in source.items()}
    return fn(source)
