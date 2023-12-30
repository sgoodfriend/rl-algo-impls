import logging
import os

import torch
from torch.optim.optimizer import StateDict

from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.algorithm import OPTIMIZER_FILENAME
from rl_algo_impls.shared.trackable import Trackable

class SynchronousAlgorithmState(Trackable):
    def __init__(self, algo: Algorithm) -> None:
        self.algo = algo

    @property
    def name(self) -> str:
        return OPTIMIZER_FILENAME

    def save(self, path: str) -> None:
        torch.save(self.get_state(), os.path.join(path, OPTIMIZER_FILENAME))

    def load(self, path: str) -> None:
        optimizer_path = os.path.join(path, OPTIMIZER_FILENAME)
        if os.path.exists(optimizer_path):
            self.set_state(torch.load(optimizer_path, map_location=self.algo.device))
        else:
            logging.info(
                f"Optimizer state not found at {optimizer_path}. Not overwriting optimizer state."
            )

    def get_state(self) -> StateDict:
        return self.algo.optimizer.state_dict()

    def set_state(self, state: StateDict) -> None:
        self.algo.optimizer.load_state_dict(state)
