# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import logging
import sys
from dataclasses import asdict
from typing import Any, Dict, List

import yaml

from rl_algo_impls.ppo.learning_rate_by_kl_divergence import LearningRateByKLDivergence
from rl_algo_impls.ppo.ppo import PPO
from rl_algo_impls.rollout.in_process_rollout import InProcessRolloutGenerator
from rl_algo_impls.rollout.reference_ai_rollout import ReferenceAIRolloutGenerator
from rl_algo_impls.rollout.sync_step_rollout import SyncStepRolloutGenerator
from rl_algo_impls.runner.config import Config, TrainArgs
from rl_algo_impls.runner.running_utils import (
    DEFAULT_ROLLOUT_GENERATORS,
    get_device,
    hparam_dict,
    initialize_policy_algo_data_store_view,
    load_hyperparams,
    set_device_optimizations,
    set_seeds,
)
from rl_algo_impls.shared.callbacks.callback import Callback
from rl_algo_impls.shared.callbacks.hyperparam_transitions import HyperparamTransitions
from rl_algo_impls.shared.callbacks.self_play_callback import SelfPlayCallback
from rl_algo_impls.shared.data_store.synchronous_data_store_accessor import (
    SynchronousDataStoreAccessor,
)
from rl_algo_impls.shared.evaluator.in_process_evaluator import InProcessEvaluator
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.summary_wrapper.in_process_summary_wrapper import (
    InProcessSummaryWrapper,
)
from rl_algo_impls.shared.summary_wrapper.remote_summary_wrapper import (
    RemoteSummaryWrapper,
)
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper
from rl_algo_impls.wrappers.vector_wrapper import find_wrapper


def train(args: TrainArgs):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info(args)
    hyperparams = load_hyperparams(args.algo, args.env)
    logging.info(hyperparams)
    config = Config(args, hyperparams, os.getcwd())

    if config.process_mode == "sync":
        tb_writer = InProcessSummaryWrapper(config, args)
    elif config.process_mode == "async":
        tb_writer = RemoteSummaryWrapper(config, args)
    else:
        raise ValueError(
            f"process_mode {config.process_mode} not recognized (Expect: sync or async)"
        )

    set_seeds(args.seed)

    data_store_accessor = SynchronousDataStoreAccessor(
        **(config.hyperparams.checkpoints_kwargs or {})
    )

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
        rollout_generator_cls = DEFAULT_ROLLOUT_GENERATORS[args.algo]

    rollout_generator = rollout_generator_cls(
        config,
        data_store_accessor,
        tb_writer,
        **rollout_hyperparams,
    )

    device = get_device(config, rollout_generator.env_spaces)
    set_device_optimizations(device, **config.device_hyperparams)
    policy, algo, learner_data_store_view = initialize_policy_algo_data_store_view(
        config,
        rollout_generator.env_spaces,
        device,
        data_store_accessor,
        tb_writer,
    )

    self_play_wrapper = (
        find_wrapper(rollout_generator.vec_env, SelfPlayWrapper)
        if isinstance(rollout_generator, InProcessRolloutGenerator)
        else None
    )
    evaluator = InProcessEvaluator(
        config,
        data_store_accessor,
        tb_writer,
        self_play_wrapper=self_play_wrapper,
        best_model_path=config.model_dir_path(best=True),
        **config.eval_callback_params(),
        video_dir=config.videos_path,
        additional_keys_to_log=config.additional_keys_to_log,
        latest_model_path=config.model_dir_path(best=False),
    )
    callbacks: List[Callback] = []
    if self_play_wrapper:
        callbacks.append(SelfPlayCallback(policy, self_play_wrapper))

    lr_by_kl_callback = None
    if config.hyperparams.lr_by_kl_kwargs:
        assert isinstance(
            algo, PPO
        ), f"lr_by_kl_kwargs only supported for PPO, not {algo.__class__.__name__}"
        lr_by_kl_callback = LearningRateByKLDivergence(
            algo,
            **config.hyperparams.lr_by_kl_kwargs,
        )
        callbacks.append(lr_by_kl_callback)
    if config.hyperparams.hyperparam_transitions_kwargs:
        callbacks.append(
            HyperparamTransitions(
                config,
                algo,
                learner_data_store_view,
                **config.hyperparams.hyperparam_transitions_kwargs,
                lr_by_kl_callback=lr_by_kl_callback,
            )
        )

    rollout_generator.prepare()
    algo.learn(
        learner_data_store_view,
        config.n_timesteps,
        callbacks=callbacks,
    )

    evaluator.save(algo.policy, config.model_dir_path(best=False))

    eval_stats = evaluator.evaluate(n_episodes=10, print_returns=True)

    log_dict: Dict[str, Any] = {
        "eval": eval_stats._asdict(),
    }
    hparam_metric_dict = {
        "hparam/last_mean": eval_stats.score.mean,
        "hparam/last_result": eval_stats.score.mean - eval_stats.score.std,
    }
    best_eval_stats = evaluator.close()
    if best_eval_stats:
        log_dict["best_eval"] = best_eval_stats._asdict()
        hparam_metric_dict.update(
            {
                "hparam/best_mean": best_eval_stats.score.mean,
                "hparam/best_result": best_eval_stats.score.mean
                - best_eval_stats.score.std,
            }
        )
    log_dict.update(asdict(hyperparams))
    log_dict.update(vars(args))
    with open(config.logs_path, "a") as f:
        yaml.dump({config.run_name(): log_dict}, f)

    tb_writer.add_hparams(
        hparam_dict(hyperparams, vars(args)),
        hparam_metric_dict,
        None,
        config.run_name(),
    )

    tb_writer.close()
