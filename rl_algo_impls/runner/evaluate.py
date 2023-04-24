import os
import shutil
from dataclasses import dataclass
from typing import NamedTuple, Optional

from rl_algo_impls.runner.config import Config, EnvHyperparams, Hyperparams, RunArgs
from rl_algo_impls.runner.running_utils import (
    get_device,
    load_hyperparams,
    make_policy,
    set_seeds,
)
from rl_algo_impls.shared.callbacks.eval_callback import evaluate
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.vec_env import make_eval_env


@dataclass
class EvalArgs(RunArgs):
    render: bool = True
    best: bool = True
    n_envs: Optional[int] = 1
    n_episodes: int = 3
    deterministic_eval: Optional[bool] = None
    no_print_returns: bool = False
    wandb_run_path: Optional[str] = None


class Evaluation(NamedTuple):
    policy: Policy
    stats: EpisodesStats
    config: Config


def evaluate_model(args: EvalArgs, root_dir: str) -> Evaluation:
    if args.wandb_run_path:
        import wandb

        api = wandb.Api()
        run = api.run(args.wandb_run_path)
        params = run.config

        args.algo = params["algo"]
        args.env = params["env"]
        args.seed = params.get("seed", None)
        args.use_deterministic_algorithms = params.get(
            "use_deterministic_algorithms", True
        )

        config = Config(args, Hyperparams.from_dict_with_extra_fields(params), root_dir)
        model_path = config.model_dir_path(best=args.best, downloaded=True)

        model_archive_name = config.model_dir_name(best=args.best, extension=".zip")
        run.file(model_archive_name).download()
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        shutil.unpack_archive(model_archive_name, model_path)
        os.remove(model_archive_name)
    else:
        hyperparams = load_hyperparams(args.algo, args.env)

        config = Config(args, hyperparams, root_dir)
        model_path = config.model_dir_path(best=args.best)

    print(args)

    set_seeds(args.seed, args.use_deterministic_algorithms)

    env = make_eval_env(
        config,
        EnvHyperparams(**config.env_hyperparams),
        override_hparams={"n_envs": args.n_envs} if args.n_envs else None,
        render=args.render,
        normalize_load_path=model_path,
    )
    device = get_device(config, env)
    policy = make_policy(
        config,
        env,
        device,
        load_path=model_path,
        **config.policy_hyperparams,
    ).eval()

    deterministic = (
        args.deterministic_eval
        if args.deterministic_eval is not None
        else config.eval_hyperparams.get("deterministic", True)
    )
    return Evaluation(
        policy,
        evaluate(
            env,
            policy,
            args.n_episodes,
            render=args.render,
            deterministic=deterministic,
            print_returns=not args.no_print_returns,
        ),
        config,
    )
