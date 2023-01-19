import numpy as np
import torch

from dataclasses import dataclass
from typing import List, NamedTuple

from shared.policy.on_policy import Critic


@dataclass
class Trajectory:
    obs: List[np.ndarray]
    act: List[np.ndarray]
    rew: List[float]
    v: List[float]
    terminated: bool

    def __init__(self) -> None:
        self.obs = []
        self.act = []
        self.rew = []
        self.v = []
        self.terminated = False

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, v: float):
        self.obs.append(obs)
        self.act.append(act)
        self.rew.append(rew)
        self.v.append(v)

    def __len__(self) -> int:
        return len(self.obs)
