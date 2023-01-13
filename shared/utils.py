import numpy as np

from typing import Callable

Schedule = Callable[[float], float]

def linear_schedule(start_val: float, end_val: float,
                    end_fraction: float) -> Schedule:

    def func(progress_fraction: float) -> float:
        if progress_fraction >= end_fraction:
            return end_val
        else:
            return (start_val +
                    (end_val - start_val) * progress_fraction / end_fraction)

    return func


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    dc = x.copy()
    for i in reversed(range(len(x) - 1)):
        dc[i] += gamma * dc[i + 1]
    return dc