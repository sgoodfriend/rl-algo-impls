from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ByteArray = np.ndarray  # [Any, np.dtype[np.int8]] (Requires Python 3.9)
MicroRTSObservation = Tuple[List[ByteArray], List[ByteArray]]


class MicroRTSInterfaceListener(ABC):
    @abstractmethod
    def map_change(
        self,
        old_heights: List[int],
        old_widths: List[int],
        old_terrains: List[Optional[np.ndarray]],
        indexes_changed: int,
    ) -> None:
        ...


class MicroRTSInterface(ABC):
    metadata = {"render.modes": []}

    @abstractmethod
    def step(
        self, action
    ) -> Tuple[List[ByteArray], List[ByteArray], np.ndarray, np.ndarray, List[Dict]]:
        ...

    @abstractmethod
    def reset(self) -> MicroRTSObservation:
        ...

    def render(self, mode: str = "human"):
        raise NotImplementedError

    @property
    @abstractmethod
    def num_envs(self) -> int:
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
    def terrain_md5(self, env_idx: int) -> Optional[str]:
        ...

    @abstractmethod
    def resources(self, env_idx: int) -> np.ndarray:
        ...

    @property
    def is_pre_game_analysis(self) -> bool:
        return False

    @property
    def pre_game_analysis_expiration_ms(self) -> int:
        return 0

    @property
    def pre_game_analysis_folder(self) -> Optional[str]:
        return None

    def set_expected_step_ms(self, expected_step_ms: int) -> None:
        pass

    @property
    def last_action(self) -> Optional[List[List[List[int]]]]:
        return None

    @abstractmethod
    def close(self, **kwargs):
        ...

    @abstractmethod
    def add_listener(self, listener: MicroRTSInterfaceListener) -> None:
        ...

    @abstractmethod
    def remove_listener(self, listener: MicroRTSInterfaceListener) -> None:
        ...

    @abstractmethod
    def debug_matrix_obs(self, env_idx: int) -> Optional[np.ndarray]:
        ...

    @abstractmethod
    def debug_matrix_mask(self, env_idx: int) -> Optional[np.ndarray]:
        ...
