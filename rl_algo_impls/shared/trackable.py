from abc import ABC, abstractmethod
from typing import Any, TypeVar

T = TypeVar("T", bound="Trackable")


class Trackable(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...

    @abstractmethod
    def get_state(self) -> Any:
        ...

    @abstractmethod
    def set_state(self, state: Any) -> None:
        ...


class UpdateTrackable(Trackable):
    @abstractmethod
    def get_update(self) -> Any:
        ...

    @abstractmethod
    def apply_update(self, update: Any) -> None:
        ...
