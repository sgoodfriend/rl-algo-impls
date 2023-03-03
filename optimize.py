import logging
import numpy as np
import optuna
import os

from dataclasses import dataclass
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Callable, Union

from a2c.optimize import sample_params as a2c_sample_params
from runner.config import Config, RunArgs, EnvHyperparams
from runner.env import make_env, make_eval_env
from runner.running_utils import (
    base_parser,
    load_hyperparams,
    set_seeds,
    get_device,
    make_policy,
    ALGOS,
    hparam_dict,
)
from shared.callbacks.optimize_callback import OptimizeCallback
from shared.stats import EpisodesStats


@dataclass
class OptimizeArgs(RunArgs):
    n_trials: int = 100
    n_jobs: int = 1
    n_startup_trials: int = 5
    n_evaluations: int = 4
    n_eval_envs: int = 8
    n_eval_episodes: int = 16
    timeout: Union[int, float, None] = None


def parse_args() -> OptimizeArgs:
    parser = base_parser()
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Maximum number of trials"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="Number of jobs to run in parallel"
    )
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=5,
        help="Stop random sampling after this number of trials",
    )
    parser.add_argument(
        "--n-evaluations",
        type=int,
        default=4,
        help="Number of evaluations during the training",
    )
    parser.add_argument(
        "--n-eval-envs",
        type=int,
        default=8,
        help="Number of envs in vectorized eval environment",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=16,
        help="Number of episodes to complete for evaluation",
    )
    parser.add_argument("--timeout", type=int, help="Seconds to timeout optimization")
    parser.set_defaults(
        algo="a2c",
        env="LunarLander-v2",
        seed=[1],
    )
    args = vars(parser.parse_args())
    if args["seed"]:
        args["seed"] = args["seed"][0]
    return OptimizeArgs(**args)


def objective_fn(args: OptimizeArgs) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        base_hyperparams = load_hyperparams(args.algo, args.env, os.getcwd())
        if args.algo == "a2c":
            hyperparams = a2c_sample_params(trial, base_hyperparams)
        else:
            raise ValueError(f"Optimizing {args.algo} isn't supported")
        config = Config(args, hyperparams, os.getcwd())

        tb_writer = SummaryWriter(config.tensorboard_summary_path)
        set_seeds(args.seed, args.use_deterministic_algorithms)

        env = make_env(
            config, EnvHyperparams(**config.env_hyperparams), tb_writer=tb_writer
        )
        device = get_device(config.device, env)
        policy = make_policy(args.algo, env, device, **config.policy_hyperparams)
        algo = ALGOS[args.algo](
            policy, env, device, tb_writer, **config.algo_hyperparams
        )

        eval_env = make_eval_env(
            config,
            EnvHyperparams(**config.env_hyperparams),
            override_n_envs=args.n_eval_envs,
        )
        callback = OptimizeCallback(
            policy,
            eval_env,
            trial,
            tb_writer,
            step_freq=config.n_timesteps // args.n_evaluations,
            n_episodes=args.n_eval_episodes,
            deterministic=config.eval_params.get("deterministic", True),
        )
        try:
            algo.learn(config.n_timesteps, callback=callback)
        except AssertionError as e:
            logging.warning(e)
            return np.nan

        if not callback.is_pruned:
            eval_stats = callback.evaluate()
            if not callback.is_pruned:
                policy.save(config.model_dir_path(best=False))
        else:
            eval_stats: EpisodesStats = callback.last_eval_stat  # type: ignore

        tb_writer.add_hparams(
            hparam_dict(hyperparams, vars(args)),
            {
                "hparam/last_mean": eval_stats.score.mean,
                "hparam/last_result": eval_stats.score.mean - eval_stats.score.std,
                "hparam/is_pruned": callback.is_pruned,
            },
            None,
            config.run_name,
        )
        tb_writer.close()

        if callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        return eval_stats.score.mean

    return objective


if __name__ == "__main__":
    args = parse_args()

    sampler = TPESampler(n_startup_trials=args.n_startup_trials)
    pruner = MedianPruner(n_startup_trials=args.n_startup_trials)
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

    try:
        study.optimize(
            objective_fn(args),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            timeout=args.timeout,
        )
    except KeyboardInterrupt:
        pass

    best = study.best_trial
    print(f"Best Trial Value: {best.value}")
    print("Attributes:")
    for key, value in best.user_attrs.items():
        print(f"  {key}: {value}")

    print(study.trials_dataframe().to_markdown(index=False))

    fig1 = plot_optimization_history(study)
    fig1.write_image("opt_history.png")
    fig2 = plot_param_importances(study)
    fig2.write_html("param_importances.png")
    