import dataclasses
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import wandb
from rl_algo_impls.runner.config import Config, EnvHyperparams, RunArgs
from rl_algo_impls.runner.evaluate import Evaluation
from rl_algo_impls.runner.running_utils import (
    get_device,
    load_hyperparams,
    make_policy,
    set_seeds,
)
from rl_algo_impls.runner.wandb_load import load_player
from rl_algo_impls.shared.callbacks.eval_callback import evaluate
from rl_algo_impls.shared.vec_env import make_eval_env
from rl_algo_impls.wrappers.vec_episode_recorder import VecEpisodeRecorder


@dataclass
class SelfplayEvalArgs(RunArgs):
    # Either wandb_run_paths or model_file_paths must have 2 elements in it.
    wandb_run_paths: List[str] = dataclasses.field(default_factory=list)
    model_file_paths: List[str] = dataclasses.field(default_factory=list)
    render: bool = False
    best: bool = True
    n_envs: int = 1
    n_episodes: int = 1
    deterministic_eval: Optional[bool] = None
    no_print_returns: bool = False
    video_path: Optional[str] = None


def selfplay_evaluate(args: SelfplayEvalArgs, root_dir: str) -> Evaluation:
    if args.wandb_run_paths:
        api = wandb.Api()
        args, config, player_1_model_path = load_player(
            api, args.wandb_run_paths[0], args, root_dir, args.best
        )
        _, _, player_2_model_path = load_player(
            api, args.wandb_run_paths[1], args, root_dir, args.best
        )
    elif args.model_file_paths:
        hyperparams = load_hyperparams(args.algo, args.env)

        config = Config(args, hyperparams, root_dir)
        player_1_model_path, player_2_model_path = args.model_file_paths
    else:
        raise ValueError("Must specify 2 wandb_run_paths or 2 model_file_paths")

    print(args)

    set_seeds(args.seed, args.use_deterministic_algorithms)

    env_make_kwargs = (
        config.eval_hyperparams.get("env_overrides", {}).get("make_kwargs", {}).copy()
    )
    env_make_kwargs["num_selfplay_envs"] = args.n_envs * 2
    env = make_eval_env(
        config,
        EnvHyperparams(**config.env_hyperparams),
        override_hparams={
            "n_envs": args.n_envs,
            "selfplay_bots": {
                player_2_model_path: args.n_envs,
            },
            "self_play_kwargs": {
                "num_old_policies": 0,
                "save_steps": np.inf,
                "swap_steps": np.inf,
                "bot_always_player_2": True,
            },
            "bots": None,
            "make_kwargs": env_make_kwargs,
        },
        render=args.render,
        normalize_load_path=player_1_model_path,
    )
    if args.video_path:
        env = VecEpisodeRecorder(
            env, args.video_path, max_video_length=18000, num_episodes=args.n_episodes
        )
    device = get_device(config, env)
    policy = make_policy(
        config,
        env,
        device,
        load_path=player_1_model_path,
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
