from abc import ABC, abstractmethod
from typing import Any, TypeVar

T = TypeVar("T", bound="Trackable")


class TrackableState(ABC):
    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...


class Trackable(ABC):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def get_state(self) -> TrackableState:
        ...

    @abstractmethod
    def set_state(self, state: TrackableState) -> None:
        ...


class UpdateTrackable(Trackable):
    @abstractmethod
    def get_update(self) -> Any:
        ...

    @abstractmethod
    def apply_update(self, update: Any) -> None:
        ...
