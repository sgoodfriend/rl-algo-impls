# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

from dataclasses import dataclass

from runner.env import make_eval_env
from runner.names import Names, RunArgs
from runner.running_utils import (
    POLICIES,
    base_parser,
    load_hyperparams,
    set_seeds,
    make_policy,
)
from shared.callbacks.eval_callback import evaluate


@dataclass
class EvalArgs(RunArgs):
    render: bool = True
    best: bool = True
    n_envs: int = 1
    n_episodes: int = 3
    deterministic: bool = True


if __name__ == "__main__":
    parser = base_parser()
    parser.add_argument("--render", default=True, type=bool)
    parser.add_argument("--best", default=True, type=bool)
    parser.add_argument("--n_envs", default=1, type=int)
    parser.add_argument("--n_episodes", default=3, type=int)
    parser.add_argument("--deterministic", default=True, type=bool)
    parser.set_defaults(algo="ppo", env="CartPole-v1")
    args = EvalArgs(**vars(parser.parse_args()))
    print(args)

    hyperparams = load_hyperparams(args.algo, args.env, os.path.dirname(__file__))
    names = Names(args, hyperparams, os.path.dirname(__file__))

    set_seeds(args.seed, args.use_deterministic_algorithms)

    model_path = names.model_dir_path(best=args.best)

    device = torch.device(hyperparams.get("device", "cpu"))
    env = make_eval_env(
        names,
        override_n_envs=args.n_envs,
        render=args.render,
        normalize_load_path=model_path,
        **hyperparams.get("env_hyperparams", {}),
    )
    policy = make_policy(
        args.algo,
        env,
        device,
        load_path=model_path,
        **hyperparams.get("policy_hyperparams", {}),
    ).eval()

    evaluate(
        env,
        policy,
        args.n_episodes,
        render=args.render,
        deterministic=args.deterministic,
    )
