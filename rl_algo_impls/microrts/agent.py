import argparse
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.backends.mkldnn

from rl_algo_impls.utils.system_info import (
    log_cpu_info,
    log_installed_libraries_info,
    log_memory_info,
)
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv

file_path = os.path.abspath(Path(__file__))
root_dir = str(Path(file_path).parent.parent.parent.absolute())
sys.path.append(root_dir)

from rl_algo_impls.microrts.map_size_policy_picker import (
    MapSizePolicyPicker,
    PickerArgs,
)
from rl_algo_impls.runner.config import Config, EnvHyperparams, RunArgs
from rl_algo_impls.runner.running_utils import get_device, load_hyperparams
from rl_algo_impls.shared.vec_env.make_env import make_eval_env
from rl_algo_impls.utils.timing import measure_time

MAX_TORCH_THREADS = 8

AGENT_ARGS_BY_TERRAIN_MD5 = {
    # "ac3b5a19643ee5816a1df17f2fadaae3": [],  # maps/NoWhereToRun9x8.xml
    # "f112aaf99e09861a5d6c6ec195130fa7": [],  # maps/DoubleGame24x24.xml
    # "ee6e75dae5051fe746a68b39112921c4": [],  # maps/BWDistantResources32x32.xml
    # "686eb7e687e50729cb134d3958d7814d": [],  # maps/BroodWar/(4)BloodBath.scmB.xml
}

AGENT_ARGS_BY_MAP_SIZE = {
    16: [
        PickerArgs(
            algo="ppo",
            env="Microrts-squnet-d16-128-BC-finetune",
            seed=1,
            best=False,
            use_paper_obs=False,
            size=16,
        )
    ],
    32: [
        PickerArgs(
            algo="ppo",
            env="Microrts-squnet-d16-128-map32-BC-finetune-A10",
            seed=1,
            best=False,
            use_paper_obs=False,
            size=32,
        )
    ],
    64: [
        PickerArgs(
            algo="ppo",
            env="Microrts-squnet-d16-128-map64-BC-finetune-A10",
            seed=1,
            best=False,
            use_paper_obs=False,
            size=64,
        )
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time_budget_ms",
        help="Milliseconds every turn is allowed to take",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--override_torch_threads",
        help="Override torch threads to this value. Ignoring other logic.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--use_best_models",
        help="Disable performance-based model selection. Always pick highest precedence model.",
        action="store_true",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    if args.verbose >= 1:
        sys.stderr = open("microrts_python_error.log", "w")
        logging.basicConfig(
            filename="microrts_python.log",
            filemode="w",
            format="%(name)s - %(levelname)s - %(message)s",
            level=logging.INFO if args.verbose == 1 else logging.DEBUG,
        )
        logging.info("Log file start")
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

    logger = logging.getLogger("Agent")
    logger.info("Command line arguments: %s", args)

    log_cpu_info()
    log_memory_info()
    log_installed_libraries_info()

    if torch.backends.mkldnn.is_available():
        logger.info("MKL-DNN (oneDNN) is available in PyTorch")
        if torch.backends.mkldnn.enabled:  # type: ignore
            logger.info("MKL-DNN (oneDNN) is enabled")
        else:
            logger.info("MKL-DNN (oneDNN) is disabled")
    else:
        logger.info("MKL-DNN (oneDNN) is not available in PyTorch")

    cur_torch_threads = torch.get_num_threads()
    num_cpus = multiprocessing.cpu_count()
    if args.override_torch_threads > 0:
        logger.info(
            f"Overriding torch num_threads from {cur_torch_threads} to {args.override_torch_threads}"
        )
        torch.set_num_threads(args.override_torch_threads)
        assert torch.get_num_threads() == args.override_torch_threads
    elif cur_torch_threads > MAX_TORCH_THREADS:
        logger.info(
            f"Reducing torch num_threads from {cur_torch_threads} to {MAX_TORCH_THREADS}"
        )
        torch.set_num_threads(MAX_TORCH_THREADS)
        assert torch.get_num_threads() == MAX_TORCH_THREADS
    elif num_cpus != cur_torch_threads:
        logger.info(
            f"Number of CPUs {num_cpus} different from PyTorch {cur_torch_threads}. Not changing"
        )
    elif cur_torch_threads > 1:
        next_lower_pow_2 = 2 ** ((cur_torch_threads - 1).bit_length() - 1)
        logger.info(
            f"{cur_torch_threads} processing units. Setting PyTorch to use {next_lower_pow_2} threads"
        )
        torch.set_num_threads(next_lower_pow_2)
        assert torch.get_num_threads() == next_lower_pow_2
    else:
        logger.info("Only 1 processing unit. Single threading.")

    run_args = RunArgs(algo="ppo", env="Microrts-agent", seed=1)
    hyperparams = load_hyperparams(run_args.algo, run_args.env)
    env_config = Config(run_args, hyperparams, root_dir)

    env = make_eval_env(
        env_config,
        EnvHyperparams(**env_config.env_hyperparams),
        override_hparams={
            "time_budget_ms": args.time_budget_ms,
        },
    )

    envs_by_name: Dict[str, VecEnv] = {}
    for sz, picker_args in AGENT_ARGS_BY_MAP_SIZE.items():
        for p_arg in picker_args:
            envs_by_name[p_arg.env] = make_eval_env(
                env_config,
                EnvHyperparams(**env_config.env_hyperparams),
                override_hparams={
                    "valid_sizes": [sz],
                    "paper_planes_sizes": [sz] if p_arg.use_paper_obs else [],
                    "fixed_size": True,
                },
            )
    for terrain_md5, picker_args in AGENT_ARGS_BY_TERRAIN_MD5.items():
        for p_arg in picker_args:
            envs_by_name[p_arg.env] = make_eval_env(
                env_config,
                EnvHyperparams(**env_config.env_hyperparams),
                override_hparams={
                    "valid_sizes": None,
                    "paper_planes_sizes": None,
                    "fixed_size": True,
                    "terrain_overrides": {
                        p_arg.map_name: {
                            "md5_hash": terrain_md5,
                            "size": p_arg.size,
                            "use_paper_obs": p_arg.use_paper_obs,
                        }
                    },
                },
            )

    device = get_device(env_config, env)
    policy = MapSizePolicyPicker(
        AGENT_ARGS_BY_MAP_SIZE,
        AGENT_ARGS_BY_TERRAIN_MD5,
        env,
        device,
        envs_by_name,
        args.time_budget_ms,
        args.use_best_models,
    ).eval()

    get_action_mask = getattr(env, "get_action_mask")

    obs = env.reset()
    action_mask = get_action_mask()
    # Runs forever. Java process expected to terminate on own.
    while True:
        if getattr(env, "is_pre_game_analysis", False):
            act = policy.pre_game_analysis(
                obs, deterministic=False, action_masks=action_mask
            )
        else:
            act_start = time.perf_counter()
            with measure_time("policy.act", threshold_ms=args.time_budget_ms):
                act = policy.act(obs, deterministic=False, action_masks=action_mask)
            act_duration = (time.perf_counter() - act_start) * 1000
            if act_duration >= args.time_budget_ms:
                logger.warn(f"act took too long: {int(act_duration)}ms")
        obs, _, _, _ = env.step(act)

        action_mask = get_action_mask()


if __name__ == "__main__":
    main()
