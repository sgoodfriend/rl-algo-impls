# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from multiprocessing import Pool

from runner.running_utils import ALGOS, base_parser
from runner.train import train, TrainArgs

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
        "--pool-size", type=int, default=1, help="Simultaneous training jobs to run"
    )
    parser.set_defaults(algo="vpg", env="CartPole-v0", seed=[1, 2, 3])
    args = parser.parse_args()
    print(args)

    # pool_size isn't a TrainArg so must be removed from args
    pool_size = args.pool_size
    delattr(args, "pool_size")

    algos = args.algo if isinstance(args.algo, list) else [args.algo]
    envs = args.env if isinstance(args.env, list) else [args.env]
    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    if all(len(arg) == 1 for arg in [algos, envs, seeds]):
        train(TrainArgs(**vars(args)))
    # Force a new process for each job to get around wandb not allowing more than one
    # wandb.tensorboard.patch call per process.
    with Pool(pool_size, maxtasksperchild=1) as p:
        train_args = []
        for algo in algos:
            for env in envs:
                for seed in seeds:
                    args_dict = vars(args).copy()
                    args_dict.update(
                        {
                            "algo": algo,
                            "env": env,
                            "seed": seed,
                        }
                    )
                    train_args.append(TrainArgs(**args_dict))
        p.map(train, train_args)
