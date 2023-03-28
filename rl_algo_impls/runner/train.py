# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import dataclasses
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence

import yaml
from torch.utils.tensorboard.writer import SummaryWriter

import wandb
from rl_algo_impls.runner.config import Config, EnvHyperparams, RunArgs
from rl_algo_impls.runner.running_utils import (
    ALGOS,
    get_device,
    hparam_dict,
    load_hyperparams,
    make_policy,
    plot_eval_callback,
    set_seeds,
)
from rl_algo_impls.shared.callbacks.eval_callback import EvalCallback
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.vec_env import make_env, make_eval_env


@dataclass
class TrainArgs(RunArgs):
    wandb_project_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: Sequence[str] = dataclasses.field(default_factory=list)
    wandb_group: Optional[str] = None


def train(args: TrainArgs):
    print(args)
    hyperparams = load_hyperparams(args.algo, args.env)
    print(hyperparams)
    config = Config(args, hyperparams, os.getcwd())

    wandb_enabled = args.wandb_project_name
    if wandb_enabled:
        wandb.tensorboard.patch(
            root_logdir=config.tensorboard_summary_path, pytorch=True
        )
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=asdict(hyperparams),
            name=config.run_name(),
            monitor_gym=True,
            save_code=True,
            tags=args.wandb_tags,
            group=args.wandb_group,
        )
        wandb.config.update(args)

    tb_writer = SummaryWriter(config.tensorboard_summary_path)

    set_seeds(args.seed, args.use_deterministic_algorithms)

    env = make_env(
        config, EnvHyperparams(**config.env_hyperparams), tb_writer=tb_writer
    )
    device = get_device(config, env)
    policy = make_policy(args.algo, env, device, **config.policy_hyperparams)
    algo = ALGOS[args.algo](policy, env, device, tb_writer, **config.algo_hyperparams)

    num_parameters = policy.num_parameters()
    num_trainable_parameters = policy.num_trainable_parameters()
    if wandb_enabled:
        wandb.run.summary["num_parameters"] = num_parameters
        wandb.run.summary["num_trainable_parameters"] = num_trainable_parameters
    else:
        print(
            f"num_parameters = {num_parameters} ; "
            f"num_trainable_parameters = {num_trainable_parameters}"
        )

    eval_env = make_eval_env(config, EnvHyperparams(**config.env_hyperparams))
    record_best_videos = config.eval_params.get("record_best_videos", True)
    callback = EvalCallback(
        policy,
        eval_env,
        tb_writer,
        best_model_path=config.model_dir_path(best=True),
        **config.eval_params,
        video_env=make_eval_env(
            config, EnvHyperparams(**config.env_hyperparams), override_n_envs=1
        )
        if record_best_videos
        else None,
        best_video_dir=config.best_videos_dir,
        additional_keys_to_log=config.additional_keys_to_log,
    )
    algo.learn(config.n_timesteps, callback=callback)

    policy.save(config.model_dir_path(best=False))

    eval_stats = callback.evaluate(n_episodes=10, print_returns=True)

    plot_eval_callback(callback, tb_writer, config.run_name())

    log_dict: Dict[str, Any] = {
        "eval": eval_stats._asdict(),
    }
    if callback.best:
        log_dict["best_eval"] = callback.best._asdict()
    log_dict.update(asdict(hyperparams))
    log_dict.update(vars(args))
    with open(config.logs_path, "a") as f:
        yaml.dump({config.run_name(): log_dict}, f)

    best_eval_stats: EpisodesStats = callback.best  # type: ignore
    tb_writer.add_hparams(
        hparam_dict(hyperparams, vars(args)),
        {
            "hparam/best_mean": best_eval_stats.score.mean,
            "hparam/best_result": best_eval_stats.score.mean
            - best_eval_stats.score.std,
            "hparam/last_mean": eval_stats.score.mean,
            "hparam/last_result": eval_stats.score.mean - eval_stats.score.std,
        },
        None,
        config.run_name(),
    )

    tb_writer.close()

    if wandb_enabled:
        shutil.make_archive(
            os.path.join(wandb.run.dir, config.model_dir_name()),
            "zip",
            config.model_dir_path(),
        )
        shutil.make_archive(
            os.path.join(wandb.run.dir, config.model_dir_name(best=True)),
            "zip",
            config.model_dir_path(best=True),
        )
        wandb.finish()
