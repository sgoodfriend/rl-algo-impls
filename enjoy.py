# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import shutil
import yaml

from dataclasses import dataclass
from typing import Optional

from runner.env import make_eval_env
from runner.config import Config, RunArgs
from runner.running_utils import (
    base_parser,
    load_hyperparams,
    set_seeds,
    get_device,
    make_policy,
)
from shared.callbacks.eval_callback import evaluate


@dataclass
class EvalArgs(RunArgs):
    render: bool = True
    best: bool = True
    n_envs: int = 1
    n_episodes: int = 3
    deterministic: Optional[bool] = None
    wandb_run_path: Optional[str] = None


if __name__ == "__main__":
    parser = base_parser()
    parser.add_argument("--render", default=True, type=bool)
    parser.add_argument("--best", default=True, type=bool)
    parser.add_argument("--n_envs", default=1, type=int)
    parser.add_argument("--n_episodes", default=3, type=int)
    parser.add_argument("--deterministic", default=None, type=bool)
    parser.add_argument("--wandb-run-path", default=None, type=str)
    parser.set_defaults(
        wandb_run_path="sgoodfriend/rl-algo-impls/sfi78a3t",
    )
    args = EvalArgs(**vars(parser.parse_args()))

    if args.wandb_run_path:
        import wandb

        api = wandb.Api()
        run = api.run(args.wandb_run_path)
        hyperparams = run.config

        args.algo = hyperparams["algo"]
        args.env = hyperparams["env"]
        args.use_deterministic_algorithms = hyperparams.get(
            "use_deterministic_algorithms", True
        )

        config = Config(args, hyperparams, os.path.dirname(__file__))
        model_path = config.model_dir_path(best=args.best, downloaded=True)

        model_archive_name = config.model_dir_name(best=args.best, extension=".zip")
        run.file(model_archive_name).download()
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        shutil.unpack_archive(model_archive_name, model_path)
        os.remove(model_archive_name)
    else:
        hyperparams = load_hyperparams(args.algo, args.env, os.path.dirname(__file__))

        config = Config(args, hyperparams, os.path.dirname(__file__))
        model_path = config.model_dir_path(best=args.best)

    print(args)

    set_seeds(args.seed, args.use_deterministic_algorithms)

    env = make_eval_env(
        config,
        override_n_envs=args.n_envs,
        render=args.render,
        normalize_load_path=model_path,
        **config.env_hyperparams,
    )
    device = get_device(config.device, env)
    policy = make_policy(
        args.algo,
        env,
        device,
        load_path=model_path,
        **config.policy_hyperparams,
    ).eval()

    if args.deterministic is None:
        deterministic = config.eval_params.get("deterministic", True)
    else:
        deterministic = args.deterministic
    evaluate(
        env,
        policy,
        args.n_episodes,
        render=args.render,
        deterministic=deterministic,
    )
