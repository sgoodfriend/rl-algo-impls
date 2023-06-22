# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import wandb
from rl_algo_impls.runner.config import RunArgs
from rl_algo_impls.runner.wandb_load import load_player

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from rl_algo_impls.runner.running_utils import base_parser


@dataclass
class DownloadArgs(RunArgs):
    best: bool = True
    wandb_run_path: Optional[str] = None


def download_model() -> None:
    parser = base_parser(multiple=False)
    parser.add_argument("--best", default=True, type=bool)
    parser.add_argument("--wandb-run-path", default=None, type=str)
    parser.set_defaults(
        algo=["ppo"],
        wandb_run_path="sgoodfriend/rl-algo-impls-benchmarks/xcmgltcu",
    )
    args = parser.parse_args()
    download_args = DownloadArgs(**vars(args))

    api = wandb.Api()
    load_player(
        api,
        args.wandb_run_path,
        download_args,
        str(Path(os.path.abspath(__file__)).parent.parent.absolute()),
        args.best,
    )
