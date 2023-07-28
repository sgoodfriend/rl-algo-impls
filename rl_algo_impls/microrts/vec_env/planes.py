import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sized, Tuple, TypeVar

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
    set_out_of_range_to_0: bool = False

    def transform(
        self,
        source: np.ndarray,
        source_col: int,
        destination: np.ndarray,
        destination_col: int,
    ) -> int:
        col = source[:, source_col]
        if self.set_out_of_range_to_0:
            col[np.logical_or(col < 0, col >= self.num_planes)] = 0
        if np.any(np.logical_or(col < 0, col >= self.num_planes)):
            logging.warn(
                f"{self.__class__.__name__}: source_col {source_col} has values outside {self.num_planes}"
            )
        destination[:, destination_col : destination_col + self.num_planes] = np.eye(
            self.num_planes
        )[col]
        return destination_col + self.num_planes

    @property
    def n_dim(self) -> int:
        return self.num_planes


@dataclass
class Planes:
    name: str
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


ObservationTransformSelf = TypeVar(
    "ObservationTransformSelf", bound="ObservationTransform"
)


class ObservationTransform(Sized):
    _planes: List[Planes]

    def __init__(
        self,
        planes: List[Planes],
        full_transform: Optional[ObservationTransformSelf] = None,
    ) -> None:
        if full_transform is not None:
            planes_by_name = {ps.name: ps for ps in planes}
            self._planes = [
                Planes(
                    ps.name,
                    planes_by_name[ps.name].planes if ps.name in planes_by_name else [],
                )
                for ps in full_transform
            ]
        else:
            self._planes = planes

        col_offset = 0
        self._planes_by_name = {}
        for ps in self._planes:
            self._planes_by_name[ps.name] = (ps, col_offset)
            col_offset += ps.n_dim

        self.n_dim = col_offset

    def planes_by_name(self, name: str) -> Planes:
        return self._planes_by_name[name][0]

    def col_offset_by_name(self, name: str) -> int:
        return self._planes_by_name[name][1]

    def __iter__(self):
        return iter(self._planes)

    def __len__(self) -> int:
        return len(self._planes)

    def append(self, planes: Planes):
        self._planes.append(planes)

        self._planes_by_name[planes.name] = (planes, self.n_dim)
        self.n_dim += planes.n_dim
