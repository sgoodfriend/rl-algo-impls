from typing import Callable

from torch.optim import Optimizer

Schedule = Callable[[float], float]


def lerp(start, end, progress):
    return start + (end - start) * progress


def linear_schedule(
    start_val: float, end_val: float, end_fraction: float = 1.0
) -> Schedule:
    def func(progress_fraction: float) -> float:
        if progress_fraction >= end_fraction:
            return end_val
        else:
            return lerp(start_val, end_val, progress_fraction / end_fraction)

    return func


def constant_schedule(val: float) -> Schedule:
    return lambda f: val


def spike_schedule(
    max_value: float,
    start_fraction: float = 1e-2,
    end_fraction: float = 1e-4,
    peak_progress: float = 0.1,
) -> Schedule:
    assert 0 < peak_progress < 1

    def func(progress_fraction: float) -> float:
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


def schedule(name: str, start_val: float) -> Schedule:
    if name == "linear":
        return linear_schedule(start_val, 0)
    elif name == "none":
        return constant_schedule(start_val)
    elif name == "spike":
        return spike_schedule(start_val)
    else:
        raise ValueError(f"Schedule {name} not supported")


def update_learning_rate(optimizer: Optimizer, learning_rate: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
