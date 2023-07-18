from enum import Enum
from typing import TypeVar, Union

import numpy as np


class InterpolateMethod(Enum):
    LINEAR = 0
    COSINE = 1


F = TypeVar("F", float, np.ndarray, Union[float, np.ndarray])


def interpolate(start: F, end: F, progress: float, method: InterpolateMethod) -> F:
    if method == InterpolateMethod.LINEAR:
        return lerp(start, end, progress)
    elif method == InterpolateMethod.COSINE:
        return cosine_interpolate(start, end, progress)
    else:
        raise ValueError(f"{method} not valid")


def lerp(start: F, end: F, progress: float) -> F:
    return start + (end - start) * progress


def cosine_interpolate(start: F, end: F, progress: float) -> F:
    return (1 - np.cos(progress * np.pi)) / 2 * (end - start) + start
