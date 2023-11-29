import logging
from typing import Optional, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

RunningMeanStdSelf = TypeVar("RunningMeanStdSelf", bound="RunningMeanStd")


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x: NDArray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            mean=self.mean,
            var=self.var,
            count=self.count,
        )

    def load(self, path: str, count_override: Optional[int] = None) -> None:
        data = np.load(path)
        self.mean = data["mean"]
        self.var = data["var"]
        self.count = data.get("count") if count_override is None else count_override

    def load_from(self: RunningMeanStdSelf, existing: RunningMeanStdSelf) -> None:
        self.mean = np.copy(existing.mean)
        self.var = np.copy(existing.var)
        self.count = np.copy(existing.count)


ExponentialMovingMeanVarSelf = TypeVar(
    "ExponentialMovingMeanVarSelf", bound="ExponentialMovingMeanVar"
)


class ExponentialMovingMeanVar:
    def __init__(
        self,
        alpha: Optional[float] = None,
        window_size: Optional[Union[int, float]] = None,
        shape: Tuple[int, ...] = (),
    ) -> None:
        assert (
            alpha is None or window_size is None
        ), f"Only one of alpha ({alpha}) or window_size ({window_size}) can be specified"
        if window_size is not None:
            alpha = 2 / (window_size + 1)
        assert alpha is not None, "Either alpha or window_size must be specified"
        assert 0 < alpha < 1, f"alpha ({alpha}) must be between 0 and 1 (exclusive)"
        self.alpha = alpha
        self.window_size = window_size if window_size is not None else (2 / alpha - 1)

        self.mean = np.zeros(shape, np.float64)
        self.squared_mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.initialized = False

    def update(self, x: NDArray) -> None:
        if not self.initialized:
            self.mean = np.mean(x, axis=0, dtype=np.float64)
            self.squared_mean = np.mean(x**2, axis=0, dtype=np.float64)
            self.var = np.var(x, axis=0, dtype=np.float64)
            self.initialized = True
            return

        weights = (
            self.alpha * ((1 - self.alpha) ** np.arange(x.shape[0] - 1, -1, -1))
        )[:, None]
        self.mean = np.sum(weights * x, axis=0) + (1 - np.sum(weights)) * self.mean
        self.squared_mean = (
            np.sum(weights * (x**2), axis=0)
            + (1 - np.sum(weights)) * self.squared_mean
        )

        self.var = self.squared_mean - self.mean**2

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            mean=self.mean,
            var=self.var,
            initialized=self.initialized,
        )

    def load(self, path: str, count_override: Optional[int] = None) -> None:
        data = np.load(path)
        self.mean = data["mean"]
        self.var = data["var"]
        self.squared_mean = self.var + self.mean**2
        self.initialized = data["initialized"].item()

    def load_from(
        self: ExponentialMovingMeanVarSelf, existing: ExponentialMovingMeanVarSelf
    ) -> None:
        self.mean = np.copy(existing.mean)
        self.var = np.copy(existing.var)
        self.initialized = np.copy(existing.initialized)


HybridMovingMeanVarSelf = TypeVar(
    "HybridMovingMeanVarSelf", bound="HybridMovingMeanVar"
)


class HybridMovingMeanVar:
    def __init__(
        self,
        alpha: Optional[float] = None,
        window_size: Optional[Union[int, float]] = None,
        shape: Tuple[int, ...] = (),
    ) -> None:
        self.rms = RunningMeanStd(shape=shape)
        self.emmv = ExponentialMovingMeanVar(
            alpha=alpha, window_size=window_size, shape=shape
        )

    @property
    def mean(self) -> NDArray:
        return (
            self.rms.mean if self.rms.count < self.emmv.window_size else self.emmv.mean
        )

    def var(self) -> NDArray:
        return self.rms.var if self.rms.count < self.emmv.window_size else self.emmv.var

    def update(self, x: NDArray) -> None:
        self.rms.update(x)
        self.emmv.update(x)

    def save(self, path: str) -> None:
        self.rms.save(path + "-rms")
        self.emmv.save(path + "-emmv")

    def load(self, path: str, count_override: Optional[int] = None) -> None:
        self.rms.load(path + "-rms", count_override=count_override)
        self.emmv.load(path + "-emmv", count_override=count_override)

    def load_from(
        self: HybridMovingMeanVarSelf, existing: HybridMovingMeanVarSelf
    ) -> None:
        self.rms.load_from(existing.rms)
        self.emmv.load_from(existing.emmv)
