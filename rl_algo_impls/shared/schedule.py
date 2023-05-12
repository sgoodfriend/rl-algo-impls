from typing import Callable, TypeVar

import numpy as np
from torch.optim import Optimizer

from rl_algo_impls.shared.tensor_utils import NumOrArray

NT = TypeVar("NT", float, np.ndarray, NumOrArray)
Schedule = Callable[[float], NT]


def lerp(start, end, progress):
    return start + (end - start) * progress


def linear_schedule(start_val: NT, end_val: NT, end_progress: float = 1.0) -> Schedule:
    def func(progress_fraction: float) -> NT:
        if progress_fraction >= end_progress:
            return end_val
        else:
            return lerp(start_val, end_val, progress_fraction / end_progress)

    return func


def constant_schedule(val: NT) -> Schedule:
    return lambda f: val


def spike_schedule(
    max_value: NT,
    start_fraction: NT = 1e-2,
    end_fraction: NT = 1e-4,
    peak_progress: float = 0.1,
) -> Schedule:
    assert 0 < peak_progress < 1

    def func(progress_fraction: float) -> NT:
        if progress_fraction < peak_progress:
            fraction = (
                start_fraction
                + (1 - start_fraction) * progress_fraction / peak_progress
            )
        else:
            fraction = 1 + (end_fraction - 1) * (progress_fraction - peak_progress) / (
                1 - peak_progress
            )
        return max_value * fraction

    return func


def schedule(name: str, start_val: NT) -> Schedule:
    if name == "linear":
        return linear_schedule(
            start_val,
            np.zeros_like(start_val) if isinstance(start_val, np.ndarray) else 0,
        )
    elif name == "none":
        return constant_schedule(start_val)
    elif name == "spike":
        return spike_schedule(start_val)
    else:
        raise ValueError(f"Schedule {name} not supported")


def update_learning_rate(optimizer: Optimizer, learning_rate: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
