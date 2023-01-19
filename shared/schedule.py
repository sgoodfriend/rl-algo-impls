from typing import Callable

Schedule = Callable[[float], float]


def linear_schedule(
    start_val: float, end_val: float, end_fraction: float = 1.0
) -> Schedule:
    def func(progress_fraction: float) -> float:
        if progress_fraction >= end_fraction:
            return end_val
        else:
            return start_val + (end_val - start_val) * progress_fraction / end_fraction

    return func


def constant_schedule(val: float) -> Schedule:
    return lambda f: val
