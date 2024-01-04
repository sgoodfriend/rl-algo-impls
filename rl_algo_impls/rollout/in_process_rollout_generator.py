from rl_algo_impls.rollout import create_synchronous_rollout_generator
from rl_algo_impls.rollout.rollout import Rollout
from rl_algo_impls.rollout.rollout_generator import RolloutGenerator
from rl_algo_impls.runner.config import Config, TrainArgs
from rl_algo_impls.shared.data_store.in_process_data_store_accessor import (
    InProcessDataStoreAccessor,
)
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces


class InProcessRolloutGenerator(RolloutGenerator):
    def __init__(
        self,
        args: TrainArgs,
        config: Config,
        data_store_accessor: InProcessDataStoreAccessor,
        tb_writer: AbstractSummaryWrapper,
    ) -> None:
        super().__init__()
        self.generator = create_synchronous_rollout_generator(
            args, config, data_store_accessor, tb_writer
        )
        data_store_accessor.rollout_generator = self

    def prepare(self) -> None:
        return self.generator.prepare()

    def rollout(self) -> Rollout:
        rollout = self.generator.rollout()
        assert rollout is not None, "Rollout is None"
        return rollout

    @property
    def env_spaces(self) -> EnvSpaces:
        return self.generator.env_spaces
