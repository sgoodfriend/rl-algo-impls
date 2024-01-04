import logging

import ray

from rl_algo_impls.rollout import create_synchronous_rollout_generator
from rl_algo_impls.runner.config import Config, TrainArgs
from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)


@ray.remote
class RolloutGeneratorActor:
    def __init__(
        self,
        args: TrainArgs,
        config: Config,
        data_store_accessor: AbstractDataStoreAccessor,
        tb_writer: AbstractSummaryWrapper,
    ) -> None:
        logging.basicConfig(level=logging.INFO, handlers=[])
        tb_writer.maybe_add_logging_handler()
        self.generator = create_synchronous_rollout_generator(
            args, config, data_store_accessor, tb_writer
        )

    def prepare(self) -> None:
        self.generator.prepare()
        while True:
            rollout = self.generator.rollout()
            if rollout is None:
                return
            self.generator.data_store_view.submit_rollout_update(rollout)

    def env_spaces(self):
        return self.generator.env_spaces
