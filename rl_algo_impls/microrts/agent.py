import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

import torch

from rl_algo_impls.utils.timing import measure_time

file_path = os.path.abspath(Path(__file__))
root_dir = str(Path(file_path).parent.parent.parent.absolute())
sys.path.append(root_dir)


from rl_algo_impls.microrts.vec_env.microrts_socket_env import (
    TIME_BUDGET_MS,
    set_connection_info,
)
from rl_algo_impls.runner.config import Config, EnvHyperparams, RunArgs
from rl_algo_impls.runner.running_utils import get_device, load_hyperparams, make_policy
from rl_algo_impls.shared.vec_env.make_env import make_eval_env

MODEL_LOAD_PATH = (
    "rai_microrts_saved_models/ppo-Microrts-selfplay-dc-phases-A10-S1-best"
)


def main():
    sys.stderr = open("microrts_python_error.log", "w")
    logging.basicConfig(
        filename="microrts_python.log",
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.info("Log file start")

    if len(sys.argv) >= 3:
        set_connection_info(int(sys.argv[1]), bool(int(sys.argv[2])))

    MAX_TORCH_THREADS = 16
    if torch.get_num_threads() > MAX_TORCH_THREADS:
        logging.info(
            f"Reducing torch num_threads from {torch.get_num_threads()} to {MAX_TORCH_THREADS}"
        )
        torch.set_num_threads(MAX_TORCH_THREADS)

    run_args = RunArgs(algo="ppo", env="Microrts-agent", seed=1)
    hyperparams = load_hyperparams(run_args.algo, run_args.env)
    config = Config(run_args, hyperparams, root_dir)

    env = make_eval_env(config, EnvHyperparams(**config.env_hyperparams))
    device = get_device(config, env)
    policy = make_policy(
        config,
        env,
        device,
        load_path=os.path.join(root_dir, MODEL_LOAD_PATH),
        **config.policy_hyperparams,
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
            logging.warn(f"act took too long: {int(act_duration)}ms")
        obs, _, d, _ = env.step(act)

        if d[0]:
            obs = env.reset()

        action_mask = get_action_mask()


if __name__ == "__main__":
    main()
