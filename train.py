# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import itertools

from argparse import Namespace
from multiprocessing import Pool
from typing import Any, Dict

from runner.running_utils import (
    base_parser,
    BASIC_ENVS,
    PYBULLET_ENVS,
    CAR_RACING_ENVS,
    ATARI_ENVS,
)
from runner.train import train, TrainArgs


def args_dict(algo: str, env: str, seed: str, args: Namespace) -> Dict[str, Any]:
    d = vars(args).copy()
    d.update(
        {
            "algo": algo,
            "env": env,
            "seed": seed,
        }
    )
    return d


if __name__ == "__main__":
    parser = base_parser()
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="rl-algo-impls",
        help="WandB project namme to upload training data to. If none, won't upload.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WanDB team of project. None uses default entity",
    )
    parser.add_argument(
        "--pool-size", type=int, default=3, help="Simultaneous training jobs to run"
    )
    parser.set_defaults(algo="ppo", env="CartPole-v1", seed=[1, 2, 3])
    args = parser.parse_args()
    print(args)

    # pool_size isn't a TrainArg so must be removed from args
    pool_size = args.pool_size
    delattr(args, "pool_size")

    algos = args.algo if isinstance(args.algo, list) else [args.algo]
    envs = args.env if isinstance(args.env, list) else [args.env]
    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    if all(len(arg) == 1 for arg in [algos, envs, seeds]):
        train(TrainArgs(**args_dict(algos[0], envs[0], seeds[0], args)))
    else:
        # Force a new process for each job to get around wandb not allowing more than one
        # wandb.tensorboard.patch call per process.
        with Pool(pool_size, maxtasksperchild=1) as p:
            train_args = [
                TrainArgs(**args_dict(algo, env, seed, args))
                for algo, env, seed in itertools.product(algos, envs, seeds)
            ]
            p.map(train, train_args)
