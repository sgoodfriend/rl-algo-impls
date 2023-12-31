from abc import ABC, abstractmethod


class AbstractSummaryWrapper(ABC):
    @abstractmethod
    def on_timesteps_elapsed(self, timesteps_elapsed: int) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def __getattr__(self, name: str):
        ...
