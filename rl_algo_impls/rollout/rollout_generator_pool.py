import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
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
        self.worker_hyperparams = config.worker_hyperparams
        seeds = (
            np.random.RandomState(args.seed).randint(
                0, np.iinfo(np.int32).max, self.worker_hyperparams.n_rollout_workers
            )
            if (
                args.seed is not None
                and self.worker_hyperparams.n_rollout_workers > 1
                and self.worker_hyperparams.different_seeds_for_rollout_workers
            )
            else None
        )
        self.generator_actors = []
        for rollout_worker_idx in range(self.worker_hyperparams.n_rollout_workers):
            if seeds is not None:
                _args = replace(args, seed=seeds[rollout_worker_idx])
                _config = replace(config, args=args)
            else:
                _args = args
                _config = config
            self.generator_actors.append(
                RolloutGeneratorActor.remote(
                    ray.get_runtime_context().current_actor,
                    _args,
                    _config,
                    data_store_accessor,
                    tb_writer,
                    rollout_worker_idx,
                )
            )
        self._is_started = False

    async def start(self) -> None:
        self._is_started = True
        for actor in self.generator_actors:
            actor.start.remote()
            await asyncio.sleep(
                self.worker_hyperparams.rollout_incremental_start_delay_seconds
            )

    def env_spaces(self):
        return ray.get(self.generator_actors[0].env_spaces.remote())

    def is_started(self) -> bool:
        return self._is_started
