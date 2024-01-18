import logging
from typing import TYPE_CHECKING

import ray

if TYPE_CHECKING:
    from rl_algo_impls.rollout.rollout_generator_pool import RolloutGeneratorPool
    from rl_algo_impls.runner.config import Config, TrainArgs
    from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
        AbstractDataStoreAccessor,
    )
    from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
        AbstractSummaryWrapper,
    )


@ray.remote(max_restarts=3)
class RolloutGeneratorActor:
    def __init__(
        self,
        rollout_generator_pool: "RolloutGeneratorPool",
        args: "TrainArgs",
        config: "Config",
        data_store_accessor: "AbstractDataStoreAccessor",
        tb_writer: "AbstractSummaryWrapper",
        rollout_worker_idx: int,
    ) -> None:
        from rl_algo_impls.utils.ray import init_ray_actor

        rollout_cuda_index = config.rollout_cuda_index(rollout_worker_idx)
        init_ray_actor(
            cuda_visible_devices=[rollout_cuda_index]
            if rollout_cuda_index is not None
            else []
        )

        from rl_algo_impls.rollout import create_synchronous_rollout_generator

        logging.basicConfig(level=logging.INFO, handlers=[])
        tb_writer.maybe_add_logging_handler()
        self.generator = create_synchronous_rollout_generator(
            args, config, data_store_accessor, tb_writer
        )

        self._is_started = False
        if ray.get_runtime_context().was_current_actor_reconstructed:
            print(f"{self.__class__.__name__} was reconstructed")
            logging.warning(f"{self.__class__.__name__} was reconstructed")
            if (
                ray.get(rollout_generator_pool.is_started.remote())
                and not self._is_started
            ):
                print(f"{self.__class__.__name__} restarting")
                logging.info(f"{self.__class__.__name__} restarting")
                self.start()

    def start(self) -> None:
        self._is_started = True
        self.generator.prepare()
        while True:
            rollout = self.generator.rollout()
            if rollout is None:
                return
            self.generator.data_store_view.submit_rollout_update(rollout)

    def env_spaces(self):
        return self.generator.env_spaces
