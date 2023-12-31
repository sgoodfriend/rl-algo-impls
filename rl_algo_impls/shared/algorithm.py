import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar

import torch
from torch.optim import Optimizer

from rl_algo_impls.shared.callbacks import Callback
from rl_algo_impls.shared.data_store.data_store_view import LearnerDataStoreView
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)

OPTIMIZER_FILENAME = "optimizer.pt"

AlgorithmSelf = TypeVar("AlgorithmSelf", bound="Algorithm")


class Algorithm(ABC):
    @abstractmethod
    def __init__(
        self,
        policy: Policy,
        device: torch.device,
        tb_writer: AbstractSummaryWrapper,
        learning_rate: float,
        optimizer: Optimizer,
        **kwargs,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.device = device
        self.tb_writer = tb_writer
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    @abstractmethod
    def learn(
        self: AlgorithmSelf,
        learner_data_store_view: LearnerDataStoreView,
        train_timesteps: int,
        callbacks: Optional[List[Callback]] = None,
    ) -> AlgorithmSelf:
        ...

    def save(self, path: str) -> None:
        torch.save(self.optimizer.state_dict(), os.path.join(path, OPTIMIZER_FILENAME))

    def load(self, path: str) -> None:
        optimizer_path = os.path.join(path, OPTIMIZER_FILENAME)
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=self.device)
            )
        else:
            logging.info(
                f"Optimizer state not found at {optimizer_path}. Not overwriting optimizer state."
            )
