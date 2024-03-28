import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, NamedTuple, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

from rl_algo_impls.shared.trackable import TrackableState, UpdateTrackable

RunningMeanStdSelf = TypeVar("RunningMeanStdSelf", bound="RunningMeanStd")

RMSSelf = TypeVar("RMSSelf", bound="RMS")


class RMSBatch(NamedTuple):
    mean: NDArray
    var: NDArray
    count: float

    @classmethod
    def from_rms(cls, rms: "RMS") -> "RMSBatch":
        return cls(rms.mean, rms.var, rms.count)


@dataclass
class RMS:
    mean: NDArray
    var: NDArray
    count: float
    normalize_axes: Tuple[int, ...]

    @classmethod
    def empty(
        cls: Type[RMSSelf],
        shape: Tuple[int, ...],
        epsilon: float,
        normalize_axes: Optional[Tuple[int, ...]] = None,
    ) -> RMSSelf:
        if normalize_axes:
            shape = tuple(
                s if i not in normalize_axes else 1 for i, s in enumerate(shape)
            )
        else:
            normalize_axes = tuple()
        # Increment axes by 1 to account for batch axis
        normalize_axes = tuple(a + 1 for a in normalize_axes)
        return cls(
            np.zeros(shape, np.float64),
            np.ones(shape, np.float64),
            epsilon,
            normalize_axes,
        )

    def update(self, x: np.ndarray) -> None:
        batch_rms = RMSBatch(
            np.mean(x, axis=(0,) + self.normalize_axes).reshape(self.mean.shape),
            np.var(x, axis=(0,) + self.normalize_axes).reshape(self.var.shape),
            x.shape[0],
        )
        self.update_from_batch(batch_rms)

    def update_from_batch(self, batch: RMSBatch) -> None:
        delta = batch.mean - self.mean
        total_count = self.count + batch.count

        self.mean = self.mean + delta * batch.count / total_count

        m_a = self.var * self.count
        m_b = batch.var * batch.count
        M2 = m_a + m_b + np.square(delta) * self.count * batch.count / total_count
        self.var = M2 / total_count
        self.count = total_count


@dataclass
class TrackableRMS(TrackableState):
    filename: str
    rms: RMS

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.savez_compressed(
            os.path.join(path, self.filename),
            **asdict(self.rms),
        )

    def load(self, path: str) -> None:
        data = np.load(os.path.join(path, self.filename))
        for k, v in data.items():
            assert hasattr(
                self.rms, k
            ), f"Unknown key {k} in {self.rms.__class__.__name__}"
            if type(getattr(self.rms, k)) == tuple:
                v = tuple(v)
            setattr(self.rms, k, v)


class RunningMeanStd(UpdateTrackable):
    def __init__(
        self,
        filename: str,
        epsilon: float = 1e-4,
        shape: Tuple[int, ...] = (),
        normalize_axes: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__(filename)
        self.filename = filename
        self.epsilon = epsilon
        self.rms = RMS.empty(shape, epsilon, normalize_axes)
        self.running = RMS.empty(shape, epsilon, normalize_axes)

    @property
    def mean(self) -> NDArray:
        return self.rms.mean

    @property
    def var(self) -> NDArray:
        return self.rms.var

    @property
    def count(self) -> float:
        return self.rms.count

    def update(self, x: NDArray) -> None:
        self.rms.update(x)
        self.running.update(x)

    def get_state(self) -> TrackableRMS:
        return TrackableRMS(self.filename, self.rms)

    def set_state(self, state: TrackableRMS) -> None:
        self.rms = state.rms

    def get_update(self) -> RMSBatch:
        running = RMSBatch.from_rms(self.running)
        self.running = RMS.empty(self.running.mean.shape, self.epsilon)
        return running

    def apply_update(self, update: RMSBatch) -> None:
        self.rms.update_from_batch(update)


EMMVSelf = TypeVar("EMMVSelf", bound="EMMV")


@dataclass
class EMMV:
    mean: NDArray
    squared_mean: NDArray
    var: NDArray
    initialized: bool

    @classmethod
    def empty(cls: Type[EMMVSelf], shape: Tuple[int, ...]) -> EMMVSelf:
        return cls(
            np.zeros(shape, np.float64),
            np.zeros(shape, np.float64),
            np.ones(shape, np.float64),
            False,
        )

    def update(self, x: NDArray, alpha: float) -> None:
        if not self.initialized:
            self.mean = np.mean(x, axis=0, dtype=np.float64)
            self.squared_mean = np.mean(x**2, axis=0, dtype=np.float64)
            self.var = np.var(x, axis=0, dtype=np.float64)
            self.initialized = True
            return

        weights = (alpha * ((1 - alpha) ** np.arange(x.shape[0] - 1, -1, -1)))[:, None]
        self.mean = np.sum(weights * x, axis=0) + (1 - np.sum(weights)) * self.mean
        self.squared_mean = (
            np.sum(weights * (x**2), axis=0) + (1 - np.sum(weights)) * self.squared_mean
        )

        self.var = self.squared_mean - self.mean**2


@dataclass
class TrackableEMMV(TrackableState):
    filename: str
    emmv: EMMV

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.savez_compressed(
            os.path.join(path, self.filename),
            **asdict(self.emmv),
        )

    def load(self, path: str) -> None:
        data = np.load(os.path.join(path, self.filename))
        for k, v in data.items():
            assert hasattr(
                self.emmv, k
            ), f"Unknown key {k} in {self.emmv.__class__.__name__}"
            setattr(self.emmv, k, v)


ExponentialMovingMeanVarSelf = TypeVar(
    "ExponentialMovingMeanVarSelf", bound="ExponentialMovingMeanVar"
)


class ExponentialMovingMeanVar(UpdateTrackable):
    def __init__(
        self,
        filename: str,
        alpha: Optional[float] = None,
        window_size: Optional[Union[int, float]] = None,
        shape: Tuple[int, ...] = (),
    ) -> None:
        super().__init__(filename)
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

        self.emmv = EMMV.empty(shape)

        self.running = []

    @property
    def mean(self) -> NDArray:
        return self.emmv.mean

    @property
    def var(self) -> NDArray:
        return self.emmv.var

    def update(self, x: NDArray) -> None:
        self.running.append(x)
        self.emmv.update(x, self.alpha)

    def get_state(self) -> TrackableEMMV:
        return TrackableEMMV(self.filename, self.emmv)

    def set_state(self, state: TrackableEMMV) -> None:
        self.emmv = state.emmv

    def get_update(self) -> NDArray:
        running = np.concatenate(self.running)
        self.running = []
        return running

    def apply_update(self, update: NDArray) -> None:
        self.emmv.update(update, self.alpha)


@dataclass
class TrackableHybridMMV(TrackableState):
    rms: TrackableRMS
    emmv: TrackableEMMV

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.rms.save(path)
        self.emmv.save(path)

    def load(self, path: str) -> None:
        self.rms.load(path)
        self.emmv.load(path)


HybridMovingMeanVarSelf = TypeVar(
    "HybridMovingMeanVarSelf", bound="HybridMovingMeanVar"
)


class HybridMovingMeanVar(UpdateTrackable):
    def __init__(
        self,
        filename: str,
        alpha: Optional[float] = None,
        window_size: Optional[Union[int, float]] = None,
        shape: Tuple[int, ...] = (),
    ) -> None:
        super().__init__(filename)
        self.rms = RunningMeanStd(filename + "-rms.npz", shape=shape)
        self.emmv = ExponentialMovingMeanVar(
            filename + "-emmv.npz",
            alpha=alpha,
            window_size=window_size,
            shape=shape,
        )

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

    def get_state(self) -> TrackableHybridMMV:
        return TrackableHybridMMV(self.rms.get_state(), self.emmv.get_state())

    def set_state(self, state: TrackableHybridMMV) -> None:
        self.rms.set_state(state.rms)
        self.emmv.set_state(state.emmv)

    def get_update(self) -> Dict[str, Any]:
        return {
            "rms": self.rms.get_update(),
            "emmv": self.emmv.get_update(),
        }

    def apply_update(self, update: Dict[str, Any]) -> None:
        self.rms.apply_update(update["rms"])
        self.emmv.apply_update(update["emmv"])
