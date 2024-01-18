from typing import TYPE_CHECKING

import ray

from rl_algo_impls.rollout.rollout_generator_actor import RolloutGeneratorActor

if TYPE_CHECKING:
    from rl_algo_impls.runner.config import Config, TrainArgs
    from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
        AbstractDataStoreAccessor,
    )
    from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
        AbstractSummaryWrapper,
    )


@ray.remote
class RolloutGeneratorPool:
    def __init__(
        self,
        args: "TrainArgs",
        config: "Config",
        data_store_accessor: "AbstractDataStoreAccessor",
        tb_writer: "AbstractSummaryWrapper",
    ) -> None:
        self.generator_actors = [
            RolloutGeneratorActor.remote(
                ray.get_runtime_context().current_actor,
                args,
                config,
                data_store_accessor,
                tb_writer,
                rollout_worker_idx,
            )
            for rollout_worker_idx in range(config.worker_hyperparams.n_rollout_workers)
        ]
        self._is_started = False

    async def start(self) -> None:
        self._is_started = True
        for actor in self.generator_actors:
            actor.start.remote()

    def env_spaces(self):
        return ray.get(self.generator_actors[0].env_spaces.remote())

    def is_started(self) -> bool:
        return self._is_started
