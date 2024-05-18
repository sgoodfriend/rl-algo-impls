import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional

from rl_algo_impls.runner.config import Config, Hyperparams, RunArgs
from rl_algo_impls.runner.env_hyperparams import EnvHyperparams
from rl_algo_impls.runner.evaluate import Evaluation
from rl_algo_impls.runner.running_utils import (
    base_parser,
    load_hyperparams,
    make_eval_policy,
    set_device_optimizations,
    set_seeds,
)
from rl_algo_impls.shared.data_store.data_store_view import EvalDataStoreView
from rl_algo_impls.shared.data_store.in_process_data_store_accessor import (
    InProcessDataStoreAccessor,
)
from rl_algo_impls.shared.evaluator.evaluator import evaluate
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces
from rl_algo_impls.shared.vec_env.make_env import make_eval_env
from rl_algo_impls.utils.device import get_device


@dataclass
class LinearProbeTrainArgs(RunArgs):
    render: bool = True
    best: bool = True
    wandb_run_path: Optional[str] = None
    override_hparams: Optional[Dict[str, Any]] = None
    n_envs: Optional[int] = 1
    deterministic_eval: Optional[bool] = None
    n_steps: int = 1_000_000
    no_print_returns: bool = False
    

def linear_probe_train() -> None:
    parser = base_parser(multiple=False)
    parser.add_argument("--render", default=True, type=bool)
    parser.add_argument("--best", default=True, type=bool)
    parser.add_argument("--n_envs", default=1, type=int)
    parser.add_argument("--deterministic-eval", default=None, type=bool)
    parser.add_argument("--n_steps", default=3, type=int)
    parser.add_argument(
        "--no-print-returns", action="store_true", help="Limit printing"
    )
    # wandb-run-path overrides base RunArgs
    parser.add_argument("--wandb-run-path", default=None, type=str)
    parser.add_argument("--override-hparams", default=None, type=str)
    parser.set_defaults(
        algo=["ppo"],
        wandb_run_path="sgoodfriend/mech-interp-rl-algo-impls/eh5nxxe2",
        render=False,
        override_hparams='{"bots":{"coacAI":1}, "map_paths": ["maps/10x10/basesTwoWorkers10x10.xml"]}',
    )
    args = parser.parse_args()
    args.algo = args.algo[0]
    args.env = args.env[0]
    args.seed = args.seed[0]
    args.override_hparams = (
        json.loads(args.override_hparams) if args.override_hparams else None
    )
    args = LinearProbeTrainArgs(**vars(args))

    train(args, os.getcwd())


def train(args: LinearProbeTrainArgs, root_dir: str) -> None:
    if args.wandb_run_path:
        import wandb

        api = wandb.Api()
        run = api.run(args.wandb_run_path)
        params = run.config

        args.algo = params["algo"]
        args.env = params["env"]
        args.seed = params.get("seed", None)

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

    set_seeds(args.seed)

    data_store_accessor = InProcessDataStoreAccessor(
        **(config.hyperparams.checkpoints_kwargs or {})
    )

    override_hparams = args.override_hparams or {}
    if args.n_envs:
        override_hparams["n_envs"] = args.n_envs
    env = make_eval_env(
        config,
        EnvHyperparams(**config.env_hyperparams),
        EvalDataStoreView(data_store_accessor, is_eval_job=True),
        override_hparams=override_hparams,
        render=args.render,
    )

    device = get_device(config, EnvSpaces.from_vec_env(env))
    set_device_optimizations(device, **config.device_hyperparams)
    policy = make_eval_policy(
        config,
        EnvSpaces.from_vec_env(env),
        device,
        data_store_accessor,
        load_path=model_path,
        **config.policy_hyperparams,
    ).eval()

    deterministic = (
        args.deterministic_eval
        if args.deterministic_eval is not None
        else config.eval_hyperparams.get("deterministic", True)
    )
    Evaluation(
        policy,
        evaluate(
            env,
            policy,
            1,
            render=args.render,
            deterministic=deterministic,
            print_returns=not args.no_print_returns,
        ),
        config,
        data_store_accessor,
    )

