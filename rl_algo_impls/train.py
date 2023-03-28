# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from multiprocessing import Pool

from rl_algo_impls.runner.running_utils import base_parser
from rl_algo_impls.runner.train import TrainArgs
from rl_algo_impls.runner.train import train as runner_train


def train() -> None:
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
    parser.add_argument(
        "--virtual-display", action="store_true", help="Use headless virtual display"
    )
    # parser.set_defaults(
    #     algo=["ppo"],
    #     env=["CartPole-v1"],
    #     seed=[10],
    #     pool_size=3,
    # )
    args = parser.parse_args()
    print(args)

    if args.virtual_display:
        from pyvirtualdisplay.display import Display

        virtual_display = Display(visible=False, size=(1400, 900))
        virtual_display.start()
    # virtual_display isn't a TrainArg so must be removed
    delattr(args, "virtual_display")

    pool_size = min(args.pool_size, len(args.seed))
    # pool_size isn't a TrainArg so must be removed from args
    delattr(args, "pool_size")

    train_args = TrainArgs.expand_from_dict(vars(args))
    if len(train_args) == 1:
        runner_train(train_args[0])
    else:
        # Force a new process for each job to get around wandb not allowing more than one
        # wandb.tensorboard.patch call per process.
        with Pool(pool_size, maxtasksperchild=1) as p:
            p.map(runner_train, train_args)


if __name__ == "__main__":
    train()
