import argparse
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

import torch
import torch.backends.mkldnn

from rl_algo_impls.utils.system_info import (
    log_cpu_info,
    log_installed_libraries_info,
    log_memory_info,
)

file_path = os.path.abspath(Path(__file__))
root_dir = str(Path(file_path).parent.parent.parent.absolute())
sys.path.append(root_dir)

from rl_algo_impls.microrts.map_size_policy_picker import (
    MapSizePolicyPicker,
    PickerArgs,
)
from rl_algo_impls.microrts.vec_env.microrts_socket_env import TIME_BUDGET_MS
from rl_algo_impls.runner.config import Config, EnvHyperparams, RunArgs
from rl_algo_impls.runner.running_utils import get_device, load_hyperparams
from rl_algo_impls.shared.vec_env.make_env import make_eval_env
from rl_algo_impls.utils.timing import measure_time

AGENT_ARGS_BY_TERRAIN_MD5 = {
    "ac3b5a19643ee5816a1df17f2fadaae3": PickerArgs(  # maps/NoWhereToRun9x8.xml
        algo="ppo",
        env="Microrts-finetuned-NoWhereToRun",
        seed=1,
        best=True,
        use_paper_obs=True,
    ),
    "f112aaf99e09861a5d6c6ec195130fa7": PickerArgs(  # maps/DoubleGame24x24.xml
        algo="ppo",
        env="Microrts-finetuned-DoubleGame-shaped",
        seed=1,
        best=True,
        use_paper_obs=True,
    ),
    "ee6e75dae5051fe746a68b39112921c4": PickerArgs(  # maps/BWDistantResources32x32.xml
        algo="ppo",
        env="Microrts-finetuned-DistantResources-shaped",
        seed=1,
        best=True,
        use_paper_obs=True,
    ),
    # "686eb7e687e50729cb134d3958d7814d": PickerArgs(  # maps/BroodWar/(4)BloodBath.scmB.xml
    #     algo="ppo",
    #     env="Microrts-finetuned-BloodBath-shaped",
    #     seed=1,
    #     best=True,
    #     use_paper_obs=True,
    # ),
}

AGENT_ARGS_BY_MAP_SIZE = {
    16: PickerArgs(
        algo="ppo",
        env="Microrts-A6000-finetuned-coac-mayari",
        seed=1,
        best=True,
        use_paper_obs=True,
    ),
    32: PickerArgs(
        algo="ppo",
        env="Microrts-squnet-map32-128ch-selfplay",
        seed=1,
        best=True,
        use_paper_obs=False,
    ),
    64: PickerArgs(
        algo="ppo",
        env="Microrts-squnet-map64-64ch-selfplay",
        seed=1,
        best=True,
        use_paper_obs=False,
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "max_torch_threads",
        help="Limit torch threads. Torch threads could be lower",
        type=int,
        nargs="?",
        default=16,
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
    if cur_torch_threads > args.max_torch_threads:
        logger.info(
            f"Reducing torch num_threads from {cur_torch_threads} to {args.max_torch_threads}"
        )
        torch.set_num_threads(args.max_torch_threads)
        assert torch.get_num_threads() == args.max_torch_threads
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

    env = make_eval_env(env_config, EnvHyperparams(**env_config.env_hyperparams))
    envs_per_size = {
        sz: make_eval_env(
            env_config,
            EnvHyperparams(**env_config.env_hyperparams),
            override_hparams={
                "valid_sizes": [sz],
                "paper_planes_sizes": [sz] if p_args.use_paper_obs else [],
                "fixed_size": True,
            },
        )
        for sz, p_args in AGENT_ARGS_BY_MAP_SIZE.items()
    }
    terrain_overrides = env_config.eval_hyperparams["env_overrides"].get(
        "terrain_overrides", {}
    )
    envs_by_terrain_md5 = {
        terrain_md5: make_eval_env(
            env_config,
            EnvHyperparams(**env_config.env_hyperparams),
            override_hparams={
                "valid_sizes": None,
                "paper_planes_sizes": None,
                "fixed_size": True,
                "terrain_overrides": {
                    n: t_override
                    for n, t_override in terrain_overrides.items()
                    if t_override["md5_hash"] == terrain_md5
                },
            },
        )
        for terrain_md5 in AGENT_ARGS_BY_TERRAIN_MD5
    }
    device = get_device(env_config, env)
    policy = MapSizePolicyPicker(
        AGENT_ARGS_BY_MAP_SIZE,
        AGENT_ARGS_BY_TERRAIN_MD5,
        env,
        device,
        envs_per_size,
        envs_by_terrain_md5,
    ).eval()

    get_action_mask = getattr(env, "get_action_mask")

    obs = env.reset()
    action_mask = get_action_mask()
    # Runs forever. Java process expected to terminate on own.
    while True:
        act_start = time.perf_counter()
        with measure_time("policy.act"):
            act = policy.act(obs, deterministic=False, action_masks=action_mask)

        act_duration = (time.perf_counter() - act_start) * 1000
        if act_duration >= TIME_BUDGET_MS:
            logger.warn(f"act took too long: {int(act_duration)}ms")
        obs, _, _, _ = env.step(act)

        action_mask = get_action_mask()


if __name__ == "__main__":
    main()
