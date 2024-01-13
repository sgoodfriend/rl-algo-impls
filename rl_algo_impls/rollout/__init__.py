from typing import Dict, Type

from rl_algo_impls.rollout.a2c_rollout import A2CRollout
from rl_algo_impls.rollout.acbc_rollout import ACBCRollout
from rl_algo_impls.rollout.ppo_rollout import PPORollout
from rl_algo_impls.rollout.reference_ai_rollout import ReferenceAIRolloutGenerator
from rl_algo_impls.rollout.rollout import Rollout
from rl_algo_impls.rollout.sync_step_rollout import SyncStepRolloutGenerator
from rl_algo_impls.rollout.synchronous_rollout_generator import (
    SynchronousRolloutGenerator,
)
from rl_algo_impls.runner.config import Config, TrainArgs
from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)

DEFAULT_IN_PROCESS_ROLLOUT_GENERATORS: Dict[str, Type[SynchronousRolloutGenerator]] = {
    # "dqn": ReplayBufferRolloutGenerator,
    "ppo": SyncStepRolloutGenerator,
    "a2c": SyncStepRolloutGenerator,
    "acbc": SyncStepRolloutGenerator,
    "appo": SyncStepRolloutGenerator,
}

ROLLOUT_CLASS_BY_ALGO: Dict[str, Type[Rollout]] = {
    "ppo": PPORollout,
    "a2c": A2CRollout,
    "acbc": ACBCRollout,
    "appo": PPORollout,
}


def create_synchronous_rollout_generator(
    args: TrainArgs,
    config: Config,
    data_store_accessor: AbstractDataStoreAccessor,
    tb_writer: AbstractSummaryWrapper,
) -> SynchronousRolloutGenerator:
    rollout_hyperparams = {**config.rollout_hyperparams}

    subaction_mask = config.policy_hyperparams.get("subaction_mask", None)
    if subaction_mask is not None:
        rollout_hyperparams["subaction_mask"] = subaction_mask
    if config.rollout_type:
        if config.rollout_type == "sync":
            rollout_generator_cls = SyncStepRolloutGenerator
        elif config.rollout_type == "reference":
            rollout_generator_cls = ReferenceAIRolloutGenerator
        elif config.rollout_type in {"guided", "guided_random"}:
            raise ValueError(f"{config.rollout_type} is not currently supported")
        else:
            raise ValueError(f"{config.rollout_type} not recognized rollout_type")
    else:
        rollout_generator_cls = DEFAULT_IN_PROCESS_ROLLOUT_GENERATORS[args.algo]

    return rollout_generator_cls(
        config,
        data_store_accessor,
        tb_writer,
        ROLLOUT_CLASS_BY_ALGO[args.algo],
        **rollout_hyperparams,
    )
