from typing import Optional

from rl_algo_impls.rollout.rollout_generator import RolloutGenerator
from rl_algo_impls.runner.config import Config, EnvHyperparams
from rl_algo_impls.shared.callbacks.summary_wrapper import SummaryWrapper
from rl_algo_impls.shared.data_store.synchronous_data_store_accessor import (
    SynchronousDataStoreAccessor,
)
from rl_algo_impls.shared.data_store.data_store_view import RolloutDataStoreView
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces
from rl_algo_impls.shared.vec_env.make_env import make_env


class InProcessRolloutGenerator(RolloutGenerator):
    def __init__(
        self,
        config: Config,
        data_store_accessor: SynchronousDataStoreAccessor,
        tb_writer: SummaryWrapper,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        data_store_accessor.rollout_generator = self
        self._data_store_view = RolloutDataStoreView(data_store_accessor)
        self.vec_env = make_env(
            config,
            EnvHyperparams(**config.env_hyperparams),
            self._data_store_view,
            tb_writer=tb_writer,
        )
        self.tb_writer = tb_writer
        self._env_spaces = EnvSpaces.from_vec_env(self.vec_env)

    @property
    def env_spaces(self) -> EnvSpaces:
        return self._env_spaces
