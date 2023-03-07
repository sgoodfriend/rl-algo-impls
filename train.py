# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from multiprocessing import Pool

from runner.running_utils import base_parser
from runner.train import train, TrainArgs


if __name__ == "__main__":
    parser = base_parser()
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="rl-algo-impls",
        help="WandB project name to upload training data to. If none, won't upload.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB team of project. None uses default entity",
    )
    parser.add_argument(
        "--wandb-tags", type=str, nargs="*", help="WandB tags to add to run"
    )
    parser.add_argument(
        "--pool-size", type=int, default=1, help="Simultaneous training jobs to run"
    )
    parser.set_defaults(
        algo="ppo",
        env="MountainCarContinuous-v0",
        seed=[1, 2, 3],
        pool_size=3,
    )
    args = parser.parse_args()
    print(args)

    if args.pool_size == 1:
        from pyvirtualdisplay.display import Display

        virtual_display = Display(visible=False, size=(1400, 900))
        virtual_display.start()

    # pool_size isn't a TrainArg so must be removed from args
    pool_size = min(args.pool_size, len(args.seed))
    delattr(args, "pool_size")

    train_args = TrainArgs.expand_from_dict(vars(args))
    if len(train_args) == 1:
        train(train_args[0])
    else:
        # Force a new process for each job to get around wandb not allowing more than one
        # wandb.tensorboard.patch call per process.
        with Pool(pool_size, maxtasksperchild=1) as p:
            p.map(train, train_args)
