from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from rl_algo_impls.rollout.rollout import Rollout
from rl_algo_impls.runner.config import Config
from rl_algo_impls.runner.env_hyperparams import EnvHyperparams
from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces


class SynchronousRolloutGenerator(ABC):
    def __init__(
        self,
        config: Config,
        data_store_accessor: AbstractDataStoreAccessor,
        tb_writer: AbstractSummaryWrapper,
        rollout_cls: Type[Rollout],
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config

        from rl_algo_impls.shared.data_store.data_store_view import RolloutDataStoreView
        from rl_algo_impls.shared.vec_env.make_env import make_env

        self.data_store_view = RolloutDataStoreView(data_store_accessor)
        self.vec_env = make_env(
            config,
            EnvHyperparams(**config.env_hyperparams),
            self.data_store_view,
            tb_writer=tb_writer,
        )
        self.tb_writer = tb_writer
        self.rollout_cls = rollout_cls
        self._env_spaces = EnvSpaces.from_vec_env(self.vec_env)

    @abstractmethod
    def prepare(self) -> None:
        ...

    @abstractmethod
    def rollout(self) -> Optional[Rollout]:
        ...

    @property
    def env_spaces(self) -> EnvSpaces:
        return self._env_spaces

    def update_rollout_params(self, rollout_params: Dict[str, Any]) -> None:
        for k, v in rollout_params.items():
            assert hasattr(
                self, k
            ), f"Expected {k} to be an attribute of {self.__class__.__name__}"
            v_type = type(getattr(self, k))
            setattr(self, k, v_type(v))
