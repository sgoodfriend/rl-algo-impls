# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

from shared.running_utils import (
    POLICIES,
    load_hyperparams,
    Names,
    set_seeds,
    make_env,
    make_policy,
)
from shared.callbacks.eval_callback import evaluate

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        default="dqn",
        type=str,
        choices=list(POLICIES.keys()),
        help="Abbreviation of algorithm used to train the policy",
    )
    parser.add_argument(
        "--env",
        default="BreakoutNoFrameskip-v4",
        type=str,
        help="Name of environment in gym",
    )
    parser.add_argument("--render", default=True, type=bool)
    parser.add_argument("--best", default=True, type=bool)
    parser.add_argument("--n_envs", default=1, type=int)
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="If specified, sets randomness seed and determinism",
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        default=True,
        type=bool,
        help="If seed set, set torch.use_deterministic_algorithms",
    )
    args = parser.parse_args()
    print(args)

    hyperparams = load_hyperparams(args.algo, args.env, os.path.dirname(__file__))
    names = Names(args.algo, args.env, hyperparams, os.path.dirname(__file__))

    if args.n_envs is not None:
        env_hyperparams = hyperparams.get("env_hyperparams", {})
        env_hyperparams["n_envs"] = args.n_envs
        if args.n_envs == 1:
            env_hyperparams["vec_env_class"] = "dummy"
        hyperparams["env_hyperparams"] = env_hyperparams

    set_seeds(args.seed, args.use_deterministic_algorithms)

    device = torch.device(hyperparams.get("device", "cpu"))
    env = make_env(
        args.env,
        args.seed,
        render=args.render,
        **hyperparams.get("env_hyperparams", {}),
    )
    policy = make_policy(
        args.algo,
        env,
        device,
        load_path=names.model_path(best=args.best),
        **hyperparams.get("policy_hyperparams", {}),
    ).eval()

    evaluate(env, policy, 3 if args.render else 100, render=args.render)
