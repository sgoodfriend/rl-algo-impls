# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import dataclasses
import shutil
import wandb
import yaml

from dataclasses import dataclass
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Any, Dict, Optional, Sequence

from shared.callbacks.eval_callback import EvalCallback
from runner.config import Config, EnvHyperparams, RunArgs
from runner.env import make_env, make_eval_env
from runner.running_utils import (
    ALGOS,
    load_hyperparams,
    set_seeds,
    get_device,
    make_policy,
    plot_eval_callback,
    flatten_hyperparameters,
)
from shared.stats import EpisodesStats


@dataclass
class TrainArgs(RunArgs):
    wandb_project_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: Sequence[str] = dataclasses.field(default_factory=list)


def train(args: TrainArgs):
    print(args)
    hyperparams = load_hyperparams(args.algo, args.env, os.getcwd())
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
            config=hyperparams,  # type: ignore
            name=config.run_name,
            monitor_gym=True,
            save_code=True,
            tags=args.wandb_tags,
        )
        wandb.config.update(args)

    tb_writer = SummaryWriter(config.tensorboard_summary_path)

    set_seeds(args.seed, args.use_deterministic_algorithms)

    env = make_env(
        config, EnvHyperparams(**config.env_hyperparams), tb_writer=tb_writer
    )
    device = get_device(config.device, env)
    policy = make_policy(args.algo, env, device, **config.policy_hyperparams)
    algo = ALGOS[args.algo](policy, env, device, tb_writer, **config.algo_hyperparams)

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
    )
    algo.learn(config.n_timesteps, callback=callback)

    policy.save(config.model_dir_path(best=False))

    eval_stats = callback.evaluate(n_episodes=10, print_returns=True)

    plot_eval_callback(callback, tb_writer, config.run_name)

    log_dict: Dict[str, Any] = {
        "eval": eval_stats._asdict(),
    }
    if callback.best:
        log_dict["best_eval"] = callback.best._asdict()
    log_dict.update(hyperparams)
    log_dict.update(vars(args))
    with open(config.logs_path, "a") as f:
        yaml.dump({config.run_name: log_dict}, f)

    best_eval_stats: EpisodesStats = callback.best  # type: ignore
    tb_writer.add_hparams(
        flatten_hyperparameters(hyperparams, vars(args)),
        {
            "hparam/best_mean": best_eval_stats.score.mean,
            "hparam/best_result": best_eval_stats.score.mean
            - best_eval_stats.score.std,
            "hparam/last_mean": eval_stats.score.mean,
            "hparam/last_result": eval_stats.score.mean - eval_stats.score.std,
        },
        None,
        config.run_name,
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
