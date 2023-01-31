# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import shutil
import yaml

from dataclasses import dataclass
from typing import Optional

from runner.env import make_eval_env
from runner.names import Names, RunArgs
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
        wandb_run_path="sgoodfriend/rl-algo-impls/vmd8rc48",
    )
    args = EvalArgs(**vars(parser.parse_args()))

    if args.wandb_run_path:
        import wandb

        config_path = "config.yaml"
        wandb.restore(config_path, run_path=args.wandb_run_path)
        with open(config_path, "r") as f:
            wandb_config = yaml.safe_load(f)
        os.remove(config_path)

        args.algo = wandb_config["algo"]["value"]
        args.env = wandb_config["env"]["value"]
        device_name = wandb_config.get("device", {}).get("value", "auto")
        env_hyperparams = wandb_config.get("env_hyperparams", {}).get("value", {})
        policy_hyperparams = wandb_config.get("policy_hyperparams", {}).get("value", {})
        eval_params = wandb_config.get("eval_params", {}).get("value", {})

        names = Names(args, env_hyperparams, os.path.dirname(__file__))
        model_path = names.model_dir_path(best=args.best, downloaded=True)

        model_archive_name = names.model_dir_name(best=args.best, extension=".zip")
        wandb.restore(
            model_archive_name,
            run_path=args.wandb_run_path,
        )
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        shutil.unpack_archive(model_archive_name, model_path)
        os.remove(model_archive_name)
    else:
        hyperparams = load_hyperparams(args.algo, args.env, os.path.dirname(__file__))

        device_name = hyperparams.get("device", "auto")
        env_hyperparams = hyperparams.get("env_hyperparams", {})
        policy_hyperparams = hyperparams.get("policy_hyperparams", {})
        eval_params = hyperparams.get("eval_params", {})

        names = Names(args, env_hyperparams, os.path.dirname(__file__))
        model_path = names.model_dir_path(best=args.best)

    print(args)

    set_seeds(args.seed, args.use_deterministic_algorithms)

    env = make_eval_env(
        names,
        override_n_envs=args.n_envs,
        render=args.render,
        normalize_load_path=model_path,
        **env_hyperparams,
    )
    device = get_device(device_name, env)
    policy = make_policy(
        args.algo,
        env,
        device,
        load_path=model_path,
        **policy_hyperparams,
    ).eval()

    if args.deterministic is None:
        deterministic = eval_params.get("deterministic", True)
    else:
        deterministic = args.deterministic
    evaluate(
        env,
        policy,
        args.n_episodes,
        render=args.render,
        deterministic=deterministic,
    )
