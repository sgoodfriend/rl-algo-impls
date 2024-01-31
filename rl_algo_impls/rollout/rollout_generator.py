from abc import ABC, abstractmethod
from typing import Any, Dict

from rl_algo_impls.rollout.rollout import Rollout
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces


class RolloutGenerator(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def prepare(self) -> None:
        ...

    @property
    @abstractmethod
    def env_spaces(self) -> EnvSpaces:
        ...
