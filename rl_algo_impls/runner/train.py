import os

# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Don't overwrite CUDA_VISIBLE_DEVICES on ray workers (https://discuss.ray.io/t/how-to-stop-ray-from-managing-cuda-visible-devices/8767/2)
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

import logging
import sys
from dataclasses import asdict
from typing import Any, Dict, List

import yaml

from rl_algo_impls.ppo.learning_rate_by_kl_divergence import (
    SUPPORTED_ALGORITHMS as LR_BY_KL_SUPPORTED_ALGORITHMS,
)
from rl_algo_impls.ppo.learning_rate_by_kl_divergence import LearningRateByKLDivergence
from rl_algo_impls.rollout.in_process_rollout_generator import InProcessRolloutGenerator
from rl_algo_impls.rollout.remote_rollout_generator import RemoteRolloutGenerator
from rl_algo_impls.runner.config import Config, TrainArgs
from rl_algo_impls.runner.running_utils import (
    hparam_dict,
    initialize_policy_algo_data_store_view,
    load_hyperparams,
    set_device_optimizations,
    set_seeds,
)
from rl_algo_impls.shared.callbacks.callback import Callback
from rl_algo_impls.shared.callbacks.hyperparam_transitions import HyperparamTransitions
from rl_algo_impls.shared.callbacks.self_play_callback import SelfPlayCallback
from rl_algo_impls.shared.data_store.in_process_data_store_accessor import (
    InProcessDataStoreAccessor,
)
from rl_algo_impls.shared.data_store.remote_data_store_accessor import (
    RemoteDataStoreAccessor,
)
from rl_algo_impls.shared.evaluator.in_process_evaluator import InProcessEvaluator
from rl_algo_impls.shared.evaluator.remote_evaluator import RemoteEvaluator
from rl_algo_impls.shared.summary_wrapper.in_process_summary_wrapper import (
    InProcessSummaryWrapper,
)
from rl_algo_impls.shared.summary_wrapper.remote_summary_wrapper import (
    RemoteSummaryWrapper,
)
from rl_algo_impls.utils.device import get_device, initialize_cuda_devices
from rl_algo_impls.utils.ray import maybe_init_ray
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper
from rl_algo_impls.wrappers.vector_wrapper import find_wrapper


def train(args: TrainArgs):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info(args)
    hyperparams = load_hyperparams(args.algo, args.env)
    logging.info(hyperparams)
    gpu_ids = initialize_cuda_devices(args, hyperparams)
    config = Config(args, hyperparams, os.getcwd(), gpu_ids=gpu_ids)
    maybe_init_ray(config)

    if config.process_mode == "sync":
        tb_writer = InProcessSummaryWrapper(config, args)
    elif config.process_mode == "async":
        tb_writer = RemoteSummaryWrapper(config, args)
    else:
        raise ValueError(
            f"process_mode {config.process_mode} not recognized (Expect: sync or async)"
        )

    set_seeds(args.seed)

    if config.process_mode == "async":
        data_store_accessor = RemoteDataStoreAccessor(
            **(config.hyperparams.checkpoints_kwargs or {})
        )
        rollout_generator = RemoteRolloutGenerator(
            args, config, data_store_accessor, tb_writer
        )
    elif config.process_mode == "sync":
        data_store_accessor = InProcessDataStoreAccessor(
            **(config.hyperparams.checkpoints_kwargs or {})
        )
        rollout_generator = InProcessRolloutGenerator(
            args, config, data_store_accessor, tb_writer
        )
    else:
        raise ValueError(
            f"process_mode {config.process_mode} not recognized (Expect: sync or async)"
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
        find_wrapper(rollout_generator.generator.vec_env, SelfPlayWrapper)
        if isinstance(rollout_generator, InProcessRolloutGenerator)
        else None
    )
    if config.process_mode == "async":
        assert self_play_wrapper is None, f"SelfPlayWrapper not supported in async mode"
        evaluator = RemoteEvaluator(
            config,
            data_store_accessor,
            tb_writer,
            best_model_path=config.model_dir_path(best=True),
            **config.eval_callback_params(),
            video_dir=config.videos_path,
            additional_keys_to_log=config.additional_keys_to_log,
            latest_model_path=config.model_dir_path(best=False),
        )
    elif config.process_mode == "sync":
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
    else:
        raise ValueError(
            f"process_mode {config.process_mode} not recognized (Expect: sync or async)"
        )
    learner_data_store_view.initialize_evaluator(evaluator)
    callbacks: List[Callback] = []
    if self_play_wrapper:
        callbacks.append(SelfPlayCallback(policy, self_play_wrapper))

    lr_by_kl_callback = None
    if config.hyperparams.lr_by_kl_kwargs:
        assert isinstance(
            algo, LR_BY_KL_SUPPORTED_ALGORITHMS
        ), f"lr_by_kl_kwargs only supported for {(c.__name__ for c in LR_BY_KL_SUPPORTED_ALGORITHMS)}, not {algo.__class__.__name__}"
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

    (best_eval_stats,) = data_store_accessor.close()
    evaluator.save(algo.policy, config.model_dir_path(best=False))

    eval_stats = evaluator.evaluate_latest_policy(
        algo, n_episodes=10, print_returns=True
    )

    log_dict: Dict[str, Any] = {
        "eval": eval_stats._asdict(),
    }
    hparam_metric_dict = {
        "hparam/last_mean": eval_stats.score.mean,
        "hparam/last_result": eval_stats.score.mean - eval_stats.score.std,
    }
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
