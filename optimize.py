import dataclasses
import gc
import logging
import numpy as np
import optuna
import os
import torch
import wandb

from dataclasses import asdict, dataclass
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Callable, NamedTuple, Optional, Sequence, Union

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
    wandb_project_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: Sequence[str] = dataclasses.field(default_factory=list)
    wandb_group: Optional[str] = None


@dataclass
class StudyArgs:
    load_study: bool
    study_name: Optional[str] = None
    storage_path: Optional[str] = None


class Args(NamedTuple):
    optimize_args: OptimizeArgs
    study_args: StudyArgs


def parse_args() -> Args:
    parser = base_parser()
    parser.add_argument(
        "--load-study",
        action="store_true",
        help="Load a preexisting study, useful for parallelization",
    )
    parser.add_argument("--study-name", type=str, help="Optuna study name")
    parser.add_argument(
        "--storage-path",
        type=str,
        help="Path of database for Optuna to persist to",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="rl-algo-impls-tuning",
        help="WandB project name to upload tuning data to. If none, won't upload",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        help="WandB team. None uses the default entity",
    )
    parser.add_argument(
        "--wandb-tags", type=str, nargs="*", help="WandB tags to add to run"
    )
    parser.add_argument(
        "--wandb-group", type=str, help="WandB group to group trials under"
    )
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
    # parser.set_defaults(
    #     algo=["a2c"],
    #     env=["LunarLander-v2"],
    #     seed=[1],
    #     n_trials=5,
    #     n_startup_trials=2,
    # )
    optimize_dict, study_dict = {}, {}
    for k, v in vars(parser.parse_args()).items():
        if k in {"load_study", "study_name", "storage_path"}:
            study_dict[k] = v
        else:
            if k in {"algo", "env", "seed"}:
                optimize_dict[k] = v[0]
            else:
                optimize_dict[k] = v

    opt_args = OptimizeArgs(**optimize_dict)
    study_args = StudyArgs(**study_dict)
    hyperparams = load_hyperparams(opt_args.algo, opt_args.env, os.getcwd())
    config = Config(opt_args, hyperparams, os.getcwd())
    if study_args.study_name is None:
        study_args.study_name = config.run_name
    if study_args.storage_path is None:
        study_args.storage_path = (
            f"sqlite:///{os.path.join(config.runs_dir, 'tuning.db')}"
        )
    opt_args.wandb_group = opt_args.wandb_group or study_args.study_name
    return Args(opt_args, study_args)


def objective_fn(args: OptimizeArgs) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        base_hyperparams = load_hyperparams(args.algo, args.env, os.getcwd())
        if args.algo == "a2c":
            hyperparams = a2c_sample_params(trial, base_hyperparams)
        else:
            raise ValueError(f"Optimizing {args.algo} isn't supported")
        config = Config(args, hyperparams, os.getcwd())

        wandb_enabled = bool(args.wandb_project_name)
        if wandb_enabled:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=asdict(hyperparams),
                name=f"{config.model_name()}-{str(trial.number)}",
                tags=args.wandb_tags,
                group=args.wandb_group,
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
                reinit=True,
            )
            wandb.config.update(args)

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

            if not callback.is_pruned:
                callback.evaluate()
                if not callback.is_pruned:
                    policy.save(config.model_dir_path(best=False))

            eval_stat: EpisodesStats = callback.last_eval_stat  # type: ignore
            train_stat: EpisodesStats = callback.last_train_stat  # type: ignore

            tb_writer.add_hparams(
                hparam_dict(hyperparams, vars(args)),
                {
                    "hparam/last_mean": eval_stat.score.mean,
                    "hparam/last_result": eval_stat.score.mean - eval_stat.score.std,
                    "hparam/train_mean": train_stat.score.mean,
                    "hparam/train_result": train_stat.score.mean - train_stat.score.std,
                    "hparam/score": callback.last_score(),
                    "hparam/is_pruned": callback.is_pruned,
                },
                None,
                config.run_name,
            )
            tb_writer.close()

            if wandb_enabled:
                wandb.run.summary["state"] = (
                    "Pruned" if callback.is_pruned else "Complete"
                )
                wandb.finish(quiet=True)

            if callback.is_pruned:
                raise optuna.exceptions.TrialPruned()

            return callback.last_score()
        except AssertionError as e:
            logging.warning(e)
            return np.nan
        finally:
            env.close()
            eval_env.close()
            gc.collect()
            torch.cuda.empty_cache()

    return objective


if __name__ == "__main__":
    from pyvirtualdisplay.display import Display

    virtual_display = Display(visible=False, size=(1400, 900))
    virtual_display.start()

    opt_args, study_args = parse_args()

    sampler = TPESampler(n_startup_trials=opt_args.n_startup_trials)
    pruner = MedianPruner(n_startup_trials=opt_args.n_startup_trials)
    if study_args.load_study:
        assert study_args.study_name
        assert study_args.storage_path
        study = optuna.load_study(
            study_name=study_args.study_name,
            storage=study_args.storage_path,
            sampler=sampler,
            pruner=pruner,
        )
    else:
        study = optuna.create_study(
            study_name=study_args.study_name,
            storage=study_args.storage_path,
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
        )

    try:
        study.optimize(
            objective_fn(opt_args),
            n_trials=opt_args.n_trials,
            n_jobs=opt_args.n_jobs,
            timeout=opt_args.timeout,
        )
    except KeyboardInterrupt:
        pass

    best = study.best_trial
    print(f"Best Trial Value: {best.value}")
    print("Attributes:")
    for key, value in list(best.params.items()) + list(best.user_attrs.items()):
        print(f"  {key}: {value}")

    df = study.trials_dataframe()
    df = df[df.state == "COMPLETE"].sort_values(by=["value"], ascending=False)
    print(df.to_markdown(index=False))

    fig1 = plot_optimization_history(study)
    fig1.write_image("opt_history.png")
    fig2 = plot_param_importances(study)
    fig2.write_image("param_importances.png")
