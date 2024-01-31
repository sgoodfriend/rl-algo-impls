import copy
import logging
import os
from typing import TypeVar

import torch
from torch.optim.optimizer import StateDict

from rl_algo_impls.shared.algorithm import OPTIMIZER_FILENAME, Algorithm
from rl_algo_impls.shared.trackable import TrackableState


class SynchronousAlgorithmState(TrackableState):
    def __init__(self, algo: Algorithm) -> None:
        self.algo = algo

    @property
    def name(self) -> str:
        return OPTIMIZER_FILENAME

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self._get_state(), os.path.join(path, OPTIMIZER_FILENAME))

    def load(self, path: str) -> None:
        optimizer_path = os.path.join(path, OPTIMIZER_FILENAME)
        if os.path.exists(optimizer_path):
            self._set_state(torch.load(optimizer_path, map_location=self.algo.device))
        else:
            logging.info(
                f"Optimizer state not found at {optimizer_path}. Not overwriting optimizer state."
            )

    def _get_state(self) -> StateDict:
        return self.algo.optimizer.state_dict()

    def _set_state(self, state: StateDict) -> None:
        self.algo.optimizer.load_state_dict(state)


T = TypeVar("T")


def _recursive_copy_to_device(item: T, device: torch.device) -> T:
    if torch.is_tensor(item):
        return item.to(device)
    elif isinstance(item, dict):
        return {k: _recursive_copy_to_device(v, device) for k, v in item.items()}
    elif isinstance(item, list):
        return [_recursive_copy_to_device(v, device) for v in item]
    else:
        return copy.copy(item)


class RemoteAlgorithmState(TrackableState):
    def __init__(self, algo: Algorithm) -> None:
        cpu_device = torch.device("cpu")
        orig_state_dict = algo.optimizer.state_dict()
        dest_state_dict = {}
        for k, v in orig_state_dict.items():
            if k == "state":
                dest_state_dict[k] = _recursive_copy_to_device(v, cpu_device)
            else:
                dest_state_dict[k] = copy.copy(v)
        self.state = dest_state_dict

    @property
    def name(self) -> str:
        return OPTIMIZER_FILENAME

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.state, os.path.join(path, OPTIMIZER_FILENAME))

    def load(self, path: str) -> None:
        self.state = torch.load(path, map_location=torch.device("cpu"))
