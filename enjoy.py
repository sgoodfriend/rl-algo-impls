# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from runner.evaluate import EvalArgs, evaluate_model
from runner.running_utils import base_parser


if __name__ == "__main__":
    parser = base_parser(multiple=False)
    parser.add_argument("--render", default=True, type=bool)
    parser.add_argument("--best", default=True, type=bool)
    parser.add_argument("--n_envs", default=1, type=int)
    parser.add_argument("--n_episodes", default=3, type=int)
    parser.add_argument("--deterministic-eval", default=None, type=bool)
    parser.add_argument(
        "--no-print-returns", action="store_true", help="Limit printing"
    )
    # wandb-run-path overrides base RunArgs
    parser.add_argument("--wandb-run-path", default=None, type=str)
    parser.set_defaults(
        algo=["ppo"],
    )
    args = parser.parse_args()
    args.algo = args.algo[0]
    args.env = args.env[0]
    args = EvalArgs(**vars(args))

    evaluate_model(args, os.path.dirname(__file__))
