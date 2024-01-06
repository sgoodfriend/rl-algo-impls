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
        self.generator_actors = [
            RolloutGeneratorActor.remote(args, config, data_store_accessor, tb_writer)
            for _ in range(
                config.hyperparams.worker_hyperparams.get("n_rollout_workers", 1)
            )
        ]

    def prepare(self) -> None:
        [actor.prepare.remote() for actor in self.generator_actors]

    @property
    def env_spaces(self):
        return ray.get(self.generator_actors[0].env_spaces.remote())
