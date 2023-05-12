from typing import List, Tuple, Union

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
