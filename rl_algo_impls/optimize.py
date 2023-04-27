import dataclasses
import gc
import inspect
import logging
import os
from dataclasses import asdict, dataclass
from typing import Callable, List, NamedTuple, Optional, Sequence, Union

import numpy as np
import optuna
import torch
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from torch.utils.tensorboard.writer import SummaryWriter

import wandb
from rl_algo_impls.a2c.optimize import sample_params as a2c_sample_params
from rl_algo_impls.runner.config import Config, EnvHyperparams, RunArgs
from rl_algo_impls.runner.running_utils import (
    ALGOS,
    base_parser,
    get_device,
    hparam_dict,
    load_hyperparams,
    make_policy,
    set_seeds,
)
from rl_algo_impls.shared.callbacks import Callback
from rl_algo_impls.shared.callbacks.lux_hyperparam_transitions import (
    LuxHyperparamTransitions,
)
from rl_algo_impls.shared.callbacks.optimize_callback import (
    Evaluation,
    OptimizeCallback,
    evaluation,
)
from rl_algo_impls.shared.callbacks.reward_decay_callback import RewardDecayCallback
from rl_algo_impls.shared.callbacks.self_play_callback import SelfPlayCallback
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.vec_env import make_env, make_eval_env
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper
from rl_algo_impls.wrappers.vectorable_wrapper import find_wrapper


@dataclass
class StudyArgs:
    load_study: bool
    study_name: Optional[str] = None
    storage_path: Optional[str] = None
    n_trials: int = 100
    n_jobs: int = 1
    n_evaluations: int = 4
    n_eval_envs: int = 8
    n_eval_episodes: int = 16
    timeout: Union[int, float, None] = None
    wandb_project_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: Sequence[str] = dataclasses.field(default_factory=list)
    wandb_group: Optional[str] = None
    virtual_display: bool = False


class Args(NamedTuple):
    train_args: Sequence[RunArgs]
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
    parser.add_argument(
        "--virtual-display", action="store_true", help="Use headless virtual display"
    )
    # parser.set_defaults(
    #     algo=["a2c"],
    #     env=["CartPole-v1"],
    #     seed=[100, 200, 300],
    #     n_trials=5,
    #     virtual_display=True,
    # )
    train_dict, study_dict = {}, {}
    for k, v in vars(parser.parse_args()).items():
        if k in inspect.signature(StudyArgs).parameters:
            study_dict[k] = v
        else:
            train_dict[k] = v

    study_args = StudyArgs(**study_dict)
    # Hyperparameter tuning across algos and envs not supported
    assert len(train_dict["algo"]) == 1
    assert len(train_dict["env"]) == 1
    train_args = RunArgs.expand_from_dict(train_dict)

    if not all((study_args.study_name, study_args.storage_path)):
        hyperparams = load_hyperparams(train_args[0].algo, train_args[0].env)
        config = Config(train_args[0], hyperparams, os.getcwd())
        if study_args.study_name is None:
            study_args.study_name = config.run_name(include_seed=False)
        if study_args.storage_path is None:
            study_args.storage_path = (
                f"sqlite:///{os.path.join(config.runs_dir, 'tuning.db')}"
            )
    # Default set group name to study name
    study_args.wandb_group = study_args.wandb_group or study_args.study_name

    return Args(train_args, study_args)


def objective_fn(
    args: Sequence[RunArgs], study_args: StudyArgs
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        if len(args) == 1:
            return simple_optimize(trial, args[0], study_args)
        else:
            return stepwise_optimize(trial, args, study_args)

    return objective


def simple_optimize(trial: optuna.Trial, args: RunArgs, study_args: StudyArgs) -> float:
    base_hyperparams = load_hyperparams(args.algo, args.env)
    base_config = Config(args, base_hyperparams, os.getcwd())
    if args.algo == "a2c":
        hyperparams = a2c_sample_params(trial, base_hyperparams, base_config)
    else:
        raise ValueError(f"Optimizing {args.algo} isn't supported")
    config = Config(args, hyperparams, os.getcwd())

    wandb_enabled = bool(study_args.wandb_project_name)
    if wandb_enabled:
        wandb.init(
            project=study_args.wandb_project_name,
            entity=study_args.wandb_entity,
            config=asdict(hyperparams),
            name=f"{config.model_name()}-{str(trial.number)}",
            tags=study_args.wandb_tags,
            group=study_args.wandb_group,
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
    device = get_device(config, env)
    policy_factory = lambda: make_policy(
        config, env, device, **config.policy_hyperparams
    )
    policy = policy_factory()
    algo = ALGOS[args.algo](policy, env, device, tb_writer, **config.algo_hyperparams)

    self_play_wrapper = find_wrapper(env, SelfPlayWrapper)
    eval_env = make_eval_env(
        config,
        EnvHyperparams(**config.env_hyperparams),
        override_hparams={"n_envs": study_args.n_eval_envs},
        self_play_wrapper=self_play_wrapper,
    )
    optimize_callback = OptimizeCallback(
        policy,
        eval_env,
        trial,
        tb_writer,
        step_freq=config.n_timesteps // study_args.n_evaluations,
        n_episodes=study_args.n_eval_episodes,
        deterministic=config.eval_hyperparams.get("deterministic", True),
    )
    callbacks: List[Callback] = [optimize_callback]
    if config.hyperparams.reward_decay_callback:
        callbacks.append(
            RewardDecayCallback(
                config,
                env,
                **(config.hyperparams.reward_decay_callback_kwargs or {}),
            )
        )
    if config.hyperparams.lux_hyperparam_transitions_kwargs:
        callbacks.append(
            LuxHyperparamTransitions(
                config,
                env,
                algo,
                **config.hyperparams.lux_hyperparam_transitions_kwargs,
            )
        )
    if self_play_wrapper:
        callbacks.append(SelfPlayCallback(policy, policy_factory, self_play_wrapper))
    try:
        algo.learn(config.n_timesteps, callbacks=callbacks)

        if not optimize_callback.is_pruned:
            optimize_callback.evaluate()
            if not optimize_callback.is_pruned:
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
                "hparam/score": optimize_callback.last_score,
                "hparam/is_pruned": optimize_callback.is_pruned,
            },
            None,
            config.run_name(),
        )
        tb_writer.close()

        if wandb_enabled:
            wandb.run.summary["state"] = (  # type: ignore
                "Pruned" if optimize_callback.is_pruned else "Complete"
            )
            wandb.finish(quiet=True)

        if optimize_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        return optimize_callback.last_score
    except AssertionError as e:
        logging.warning(e)
        return np.nan
    finally:
        env.close()
        eval_env.close()
        gc.collect()
        torch.cuda.empty_cache()


def stepwise_optimize(
    trial: optuna.Trial, args: Sequence[RunArgs], study_args: StudyArgs
) -> float:
    algo = args[0].algo
    env_id = args[0].env
    base_hyperparams = load_hyperparams(algo, env_id)
    base_config = Config(args[0], base_hyperparams, os.getcwd())
    if algo == "a2c":
        hyperparams = a2c_sample_params(trial, base_hyperparams, base_config)
    else:
        raise ValueError(f"Optimizing {algo} isn't supported")

    wandb_enabled = bool(study_args.wandb_project_name)
    if wandb_enabled:
        wandb.init(
            project=study_args.wandb_project_name,
            entity=study_args.wandb_entity,
            config=asdict(hyperparams),
            name=f"{str(trial.number)}-S{base_config.seed()}",
            tags=study_args.wandb_tags,
            group=study_args.wandb_group,
            save_code=True,
            reinit=True,
        )

    score = -np.inf

    for i in range(study_args.n_evaluations):
        evaluations: List[Evaluation] = []

        for arg in args:
            config = Config(arg, hyperparams, os.getcwd())

            tb_writer = SummaryWriter(config.tensorboard_summary_path)
            set_seeds(arg.seed, arg.use_deterministic_algorithms)

            env = make_env(
                config,
                EnvHyperparams(**config.env_hyperparams),
                normalize_load_path=config.model_dir_path() if i > 0 else None,
                tb_writer=tb_writer,
            )
            device = get_device(config, env)
            policy_factory = lambda: make_policy(
                config, env, device, **config.policy_hyperparams
            )
            policy = policy_factory()
            if i > 0:
                policy.load(config.model_dir_path())
            algo = ALGOS[arg.algo](
                policy, env, device, tb_writer, **config.algo_hyperparams
            )

            self_play_wrapper = find_wrapper(env, SelfPlayWrapper)
            eval_env = make_eval_env(
                config,
                EnvHyperparams(**config.env_hyperparams),
                normalize_load_path=config.model_dir_path() if i > 0 else None,
                override_hparams={"n_envs": study_args.n_eval_envs},
                self_play_wrapper=self_play_wrapper,
            )

            start_timesteps = int(i * config.n_timesteps / study_args.n_evaluations)
            train_timesteps = (
                int((i + 1) * config.n_timesteps / study_args.n_evaluations)
                - start_timesteps
            )

            callbacks = []
            if config.hyperparams.reward_decay_callback:
                callbacks.append(
                    RewardDecayCallback(
                        config,
                        env,
                        start_timesteps=start_timesteps,
                        **(config.hyperparams.reward_decay_callback_kwargs or {}),
                    )
                )
            if config.hyperparams.lux_hyperparam_transitions_kwargs:
                callbacks.append(
                    LuxHyperparamTransitions(
                        config,
                        env,
                        algo,
                        start_timesteps=start_timesteps,
                        **config.hyperparams.lux_hyperparam_transitions_kwargs,
                    )
                )
            if self_play_wrapper:
                callbacks.append(
                    SelfPlayCallback(policy, policy_factory, self_play_wrapper)
                )
            try:
                algo.learn(
                    train_timesteps,
                    callbacks=callbacks,
                    total_timesteps=config.n_timesteps,
                    start_timesteps=start_timesteps,
                )

                evaluations.append(
                    evaluation(
                        policy,
                        eval_env,
                        tb_writer,
                        study_args.n_eval_episodes,
                        config.eval_hyperparams.get("deterministic", True),
                        start_timesteps + train_timesteps,
                    )
                )

                policy.save(config.model_dir_path())

                tb_writer.close()

            except AssertionError as e:
                logging.warning(e)
                if wandb_enabled:
                    wandb_finish("Error")
                return np.nan
            finally:
                env.close()
                eval_env.close()
                gc.collect()
                torch.cuda.empty_cache()

        d = {}
        for idx, e in enumerate(evaluations):
            d[f"{idx}/eval_mean"] = e.eval_stat.score.mean
            d[f"{idx}/train_mean"] = e.train_stat.score.mean
            d[f"{idx}/score"] = e.score
        d["eval"] = np.mean([e.eval_stat.score.mean for e in evaluations]).item()
        d["train"] = np.mean([e.train_stat.score.mean for e in evaluations]).item()
        score = np.mean([e.score for e in evaluations]).item()
        d["score"] = score

        step = i + 1
        wandb.log(d, step=step)

        print(f"Trial #{trial.number} Step {step} Score: {round(score, 2)}")
        trial.report(score, step)
        if trial.should_prune():
            if wandb_enabled:
                wandb_finish("Pruned")
            raise optuna.exceptions.TrialPruned()

    if wandb_enabled:
        wandb_finish("Complete")
    return score


def wandb_finish(state: str) -> None:
    wandb.run.summary["state"] = state  # type: ignore
    wandb.finish(quiet=True)


def optimize() -> None:
    from pyvirtualdisplay.display import Display

    train_args, study_args = parse_args()
    if study_args.virtual_display:
        virtual_display = Display(visible=False, size=(1400, 900))
        virtual_display.start()

    sampler = TPESampler(**TPESampler.hyperopt_parameters())
    pruner = HyperbandPruner()
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
            objective_fn(train_args, study_args),
            n_trials=study_args.n_trials,
            n_jobs=study_args.n_jobs,
            timeout=study_args.timeout,
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


if __name__ == "__main__":
    optimize()
