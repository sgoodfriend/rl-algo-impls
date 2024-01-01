from abc import ABC, abstractmethod
from typing import Any, Dict


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

    @abstractmethod
    def make_wandb_archive(self, path: str) -> None:
        ...

    @abstractmethod
    def update_summary(self, summary_update: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def log_video(self, video_path: str, fps: int) -> None:
        ...
