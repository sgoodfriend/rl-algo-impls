import ray

from rl_algo_impls.rollout.rollout_generator import RolloutGenerator
from rl_algo_impls.rollout.rollout_generator_pool import RolloutGeneratorPool
from rl_algo_impls.runner.config import Config, TrainArgs
from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)


class RemoteRolloutGenerator(RolloutGenerator):
    def __init__(
        self,
        args: TrainArgs,
        config: Config,
        data_store_accessor: AbstractDataStoreAccessor,
        tb_writer: AbstractSummaryWrapper,
    ) -> None:
        super().__init__()
        self.generator_pool = RolloutGeneratorPool.remote(
            args, config, data_store_accessor, tb_writer
        )
        self._env_spaces = None

    def prepare(self) -> None:
        self.generator_pool.start.remote()

    @property
    def env_spaces(self):
        if self._env_spaces is None:
            self._env_spaces = ray.get(self.generator_pool.env_spaces.remote())
        return self._env_spaces
