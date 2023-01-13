# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import yaml

from torch.utils.tensorboard.writer import SummaryWriter
from typing import Any, Dict

from shared.callbacks.eval_callback import EvalCallback
from shared.running_utils import (
    ALGOS,
    load_hyperparams,
    Names,
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
        default="dqn",
        type=str,
        choices=list(ALGOS.keys()),
        help="Abbreviation of algorithm for training",
    )
    parser.add_argument(
        "--env", default="CartPole-v0", type=str, help="Name of environment in gym"
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="If specified, sets randomness seed and determinism",
    )
    args = parser.parse_args()
    print(args)

    hyperparams = load_hyperparams(args.algo, args.env, os.path.dirname(__file__))
    names = Names(args.algo, args.env, hyperparams, os.path.dirname(__file__))

    tb_writer = SummaryWriter(names.tensorboard_summary_path)

    device = torch.device(hyperparams.get("device", "cpu"))
    env = make_env(args.env, **hyperparams.get("env_hyperparams", {}))
    policy = make_policy(
        args.algo, env, device, **hyperparams.get("policy_hyperparams", {})
    )

    algo = ALGOS[args.algo](
        policy, env, device, **hyperparams.get("algo_hyperparams", {})
    )
    callback = EvalCallback(
        policy,
        make_env(args.env, **hyperparams.get("env_hyperparams", {})),
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
