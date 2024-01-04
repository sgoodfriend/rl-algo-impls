import logging
import os
from typing import Any, Dict, Optional, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

from rl_algo_impls.shared.trackable import Trackable

RunningMeanStdSelf = TypeVar("RunningMeanStdSelf", bound="RunningMeanStd")


class RunningMeanStd(Trackable):
    def __init__(
        self, filename: str, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()
    ) -> None:
        self.filename = filename
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x: NDArray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    @property
    def name(self) -> str:
        return self.filename

    def save(self, path: str) -> None:
        np.savez_compressed(
            os.path.join(path, self.filename),
            **self.get_state(),
        )

    def load(self, path: str) -> None:
        data = np.load(os.path.join(path, self.filename))
        self.set_state(data)

    def get_state(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def set_state(self, state: Any) -> None:
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state.get("count")


ExponentialMovingMeanVarSelf = TypeVar(
    "ExponentialMovingMeanVarSelf", bound="ExponentialMovingMeanVar"
)


class ExponentialMovingMeanVar(Trackable):
    def __init__(
        self,
        filename: str,
        alpha: Optional[float] = None,
        window_size: Optional[Union[int, float]] = None,
        shape: Tuple[int, ...] = (),
    ) -> None:
        self.filename = filename
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

    @property
    def name(self) -> str:
        return self.filename

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
            os.path.join(path, self.filename),
            **self.get_state(),
        )

    def load(self, path: str) -> None:
        data = np.load(os.path.join(path, self.filename))
        self.set_state(data)

    def get_state(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "var": self.var,
            "initialized": np.array(self.initialized),
        }

    def set_state(self, state: Any) -> None:
        self.mean = state["mean"]
        self.var = state["var"]
        self.squared_mean = self.var + self.mean**2
        self.initialized = state["initialized"].item()


HybridMovingMeanVarSelf = TypeVar(
    "HybridMovingMeanVarSelf", bound="HybridMovingMeanVar"
)


class HybridMovingMeanVar(Trackable):
    def __init__(
        self,
        filename: str,
        alpha: Optional[float] = None,
        window_size: Optional[Union[int, float]] = None,
        shape: Tuple[int, ...] = (),
    ) -> None:
        self.filename = filename
        self.rms = RunningMeanStd(self.filename + "-rms.npz", shape=shape)
        self.emmv = ExponentialMovingMeanVar(
            self.filename + "-emmv.npz",
            alpha=alpha,
            window_size=window_size,
            shape=shape,
        )

    @property
    def name(self) -> str:
        return self.filename

    @property
    def mean(self) -> NDArray:
        emmv_frac = self.rms.count / self.emmv.window_size
        if emmv_frac >= 1:
            return self.emmv.mean
        else:
            return self.rms.mean * (1 - emmv_frac) + self.emmv.mean * emmv_frac

    @property
    def var(self) -> NDArray:
        emmv_frac = self.rms.count / self.emmv.window_size
        if emmv_frac >= 1:
            return self.emmv.var
        else:
            return self.rms.var * (1 - emmv_frac) + self.emmv.var * emmv_frac

    def update(self, x: NDArray) -> None:
        self.rms.update(x)
        self.emmv.update(x)

    def save(self, path: str) -> None:
        self.rms.save(path)
        self.emmv.save(path)

    def load(self, path: str) -> None:
        self.rms.load(path)
        self.emmv.load(path)

    def get_state(self) -> Dict[str, Any]:
        return {
            "rms": self.rms.get_state(),
            "emmv": self.emmv.get_state(),
        }

    def set_state(self, state: Any) -> None:
        self.rms.set_state(state["rms"])
        self.emmv.set_state(state["emmv"])
