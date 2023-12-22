from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar("T", bound="Trackable")


class Trackable(ABC):
    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...

    def sync(self: T, other: T) -> None:
        pass
