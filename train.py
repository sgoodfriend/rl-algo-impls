# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import wandb
import yaml

from torch.utils.tensorboard.writer import SummaryWriter
from typing import Any, Dict

from shared.callbacks.eval_callback import EvalCallback
from shared.running_utils import (
    ALGOS,
    load_hyperparams,
    Names,
    set_seeds,
    make_env,
    make_policy,
    plot_training,
    plot_eval_callback,
    flatten_hyperparameters,
)
from shared.stats import EpisodesStats

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        default="vpg",
        type=str,
        choices=list(ALGOS.keys()),
        help="Abbreviation of algorithm for training",
    )
    parser.add_argument(
        "--env",
        default="CartPole-v0",
        type=str,
        help="Name of environment in gym",
    )
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
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="rl-algo-impls",
        help="WandB project namme to upload training data to. If none, won't upload.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WanDB team of project. None uses default entity",
    )
    args = parser.parse_args()
    print(args)

    hyperparams = load_hyperparams(args.algo, args.env, os.path.dirname(__file__))
    env_hyperparams = hyperparams.get("env_hyperparams", {})
    names = Names(args.algo, args.env, hyperparams, os.path.dirname(__file__))

    wandb_enabled = args.wandb_project_name
    if wandb_enabled:
        wandb.tensorboard.patch(
            root_logdir=names.tensorboard_summary_path, pytorch=True
        )
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=hyperparams,  # type: ignore
            name=names.run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.config.update(args)

    tb_writer = SummaryWriter(names.tensorboard_summary_path)

    set_seeds(args.seed, args.use_deterministic_algorithms)

    device = torch.device(hyperparams.get("device", "cpu"))
    env = make_env(args.env, args.seed, **env_hyperparams)
    policy = make_policy(
        args.algo, env, device, **hyperparams.get("policy_hyperparams", {})
    )

    algo = ALGOS[args.algo](
        policy, env, device, tb_writer, **hyperparams.get("algo_hyperparams", {})
    )

    env_seed = args.seed
    if env_seed is not None:
        env_seed += env_hyperparams.get("n_envs", 1)
    callback = EvalCallback(
        policy,
        make_env(args.env, env_seed, **hyperparams.get("env_hyperparams", {})),
        tb_writer,
        best_model_path=names.model_path(best=True),
        **hyperparams.get("eval_params", {}),
    )
    history = algo.learn(
        int(hyperparams.get("n_timesteps", 100_000)), callback=callback
    )

    policy.save(names.model_path(best=False))

    eval_stats = callback.evaluate(n_episodes=10, print_returns=True)

    plot_training(history, tb_writer)
    plot_eval_callback(callback, tb_writer)

    log_dict: Dict[str, Any] = {
        "eval": eval_stats._asdict(),
    }
    if callback.best:
        log_dict["best_eval"] = callback.best._asdict()
    log_dict.update(hyperparams)
    log_dict.update(vars(args))
    with open(names.logs_path, "a") as f:
        yaml.dump({names.run_name: log_dict}, f)

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
        names.run_name,
    )

    if wandb_enabled:
        wandb.finish()
