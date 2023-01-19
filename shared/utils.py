import numpy as np


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    dc = x.copy()
    for i in reversed(range(len(x) - 1)):
        dc[i] += gamma * dc[i + 1]
    return dc
