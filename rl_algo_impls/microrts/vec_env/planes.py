import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


class Plane(ABC):
    @abstractmethod
    def transform(
        self,
        source: np.ndarray,
        source_col: int,
        destination: np.ndarray,
        destination_col: int,
    ) -> int:
        ...

    @property
    @abstractmethod
    def n_dim(self) -> int:
        ...


@dataclass
class OffsetPlane(Plane):
    multiplier: float
    offset: int
    clip_expected: bool = False

    def transform(
        self,
        source: np.ndarray,
        source_col: int,
        destination: np.ndarray,
        destination_col: int,
    ) -> int:
        col = (
            source[:, source_col].astype(destination.dtype) + self.offset
        ) * self.multiplier
        if not self.clip_expected and np.any(np.logical_or(col < 0, col > 1)):
            logging.warn(
                f"{self.__class__.__name__}: source_col {source_col} has scaled values outside [0, 1]"
            )
        destination[:, destination_col] = col.clip(0, 1)
        return destination_col + 1

    @property
    def n_dim(self) -> int:
        return 1


class MultiplierPlane(OffsetPlane):
    def __init__(self, multiplier: float, clip_expected: bool = False) -> None:
        super().__init__(multiplier=multiplier, offset=0, clip_expected=clip_expected)


@dataclass
class OffsetThresholdPlane(Plane):
    offset: int
    min_threshold: Optional[int]
    max_threshold: Optional[int] = None

    def transform(
        self,
        source: np.ndarray,
        source_col: int,
        destination: np.ndarray,
        destination_col: int,
    ) -> int:
        col = source[:, source_col].astype(destination.dtype) + self.offset
        if self.min_threshold is not None and self.max_threshold is not None:
            col = np.logical_and(col >= self.min_threshold, col <= self.max_threshold)
        elif self.min_threshold is not None:
            col = col >= self.min_threshold
        elif self.max_threshold is not None:
            col = col <= self.max_threshold
        else:
            raise ValueError(
                f"{self.__class__.__name__} must have min_threshold or max_threshold set"
            )
        destination[:, destination_col] = col
        return destination_col + 1

    @property
    def n_dim(self) -> int:
        return 1


class ThresholdPlane(OffsetThresholdPlane):
    def __init__(
        self, min_threshold: Optional[int], max_threshold: Optional[int] = None
    ) -> None:
        return super().__init__(
            offset=0, min_threshold=min_threshold, max_threshold=max_threshold
        )


@dataclass
class OneHotPlane(Plane):
    num_planes: int

    def transform(
        self,
        source: np.ndarray,
        source_col: int,
        destination: np.ndarray,
        destination_col: int,
    ) -> int:
        col = source[:, source_col]
        if np.any(np.logical_or(col < 0, col > self.num_planes)):
            logging.warn(
                f"{self.__class__.__name__}: source_col {source_col} has values outside {self.num_planes}"
            )
        destination[:, destination_col : destination_col + self.num_planes] = np.eye(
            self.num_planes
        )[col.clip(0, self.num_planes)]
        return destination_col + self.num_planes

    @property
    def n_dim(self) -> int:
        return self.num_planes


@dataclass
class Planes:
    planes: List[Plane]

    def transform(
        self,
        source: np.ndarray,
        source_col: int,
        destination: np.ndarray,
        destination_col: int,
    ) -> int:
        for p in self.planes:
            destination_col = p.transform(
                source, source_col, destination, destination_col
            )
        return destination_col

    @property
    def n_dim(self) -> int:
        return sum(p.n_dim for p in self.planes)
