from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ByteArray = np.ndarray  # [Any, np.dtype[np.int8]] (Requires Python 3.9)


class MicroRTSInterface(ABC):
    metadata = {"render.modes": []}

    @abstractmethod
    def step(
        self, action
    ) -> Tuple[List[ByteArray], List[ByteArray], np.ndarray, np.ndarray, List[Dict]]:
        ...

    @abstractmethod
    def reset(self) -> Tuple[List[ByteArray], List[ByteArray]]:
        ...

    def render(self, mode: str = "human"):
        raise NotImplementedError

    @property
    @abstractmethod
    def num_envs(self):
        ...

    @property
    @abstractmethod
    def heights(self) -> List[int]:
        ...

    @property
    @abstractmethod
    def widths(self) -> List[int]:
        ...

    @property
    @abstractmethod
    def utt(self) -> Dict[str, Any]:
        ...

    @property
    @abstractmethod
    def partial_obs(self) -> bool:
        ...

    @abstractmethod
    def terrain(self, env_idx: int) -> np.ndarray:
        ...

    @abstractmethod
    def resources(self, env_idx: int) -> np.ndarray:
        ...

    @abstractmethod
    def close(self, **kwargs):
        ...

    @abstractmethod
    def debug_matrix_obs(self, env_idx: int) -> Optional[np.ndarray]:
        ...

    @abstractmethod
    def debug_matrix_mask(self, env_idx: int) -> Optional[np.ndarray]:
        ...
