import ray

from rl_algo_impls.rollout.rollout_generator import RolloutGenerator
from rl_algo_impls.rollout.rollout_generator_actor import RolloutGeneratorActor
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
        self.generator_actor = RolloutGeneratorActor.remote(
            args, config, data_store_accessor, tb_writer
        )

    def prepare(self) -> None:
        self.generator_actor.prepare.remote()

    @property
    def env_spaces(self):
        return ray.get(self.generator_actor.env_spaces.remote())
