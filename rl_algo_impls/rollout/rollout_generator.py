from abc import ABC, abstractmethod
from typing import Any, Dict

from rl_algo_impls.rollout.rollout import Rollout
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces


class RolloutGenerator(ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def prepare(self) -> None:
        ...

    @abstractmethod
    def rollout(self) -> Rollout:
        ...

    @property
    @abstractmethod
    def env_spaces(self) -> EnvSpaces:
        ...

    def update_rollout_params(self, rollout_params: Dict[str, Any]) -> None:
        for k, v in rollout_params.items():
            assert hasattr(
                self, k
            ), f"Expected {k} to be an attribute of {self.__class__.__name__}"
            v_type = type(getattr(self, k))
            setattr(self, k, v_type(v))
