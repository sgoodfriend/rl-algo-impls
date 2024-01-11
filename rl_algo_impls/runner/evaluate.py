import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.rollout.rollout import flatten_actions_to_tensor, flatten_to_tensor
from rl_algo_impls.runner.config import Config, EnvHyperparams, Hyperparams, RunArgs
from rl_algo_impls.shared.vec_env.device import get_device
from rl_algo_impls.runner.running_utils import (
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
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.tensor_utils import batch_dict_keys
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces
from rl_algo_impls.shared.vec_env.make_env import make_eval_env
from rl_algo_impls.wrappers.vec_episode_recorder import VecEpisodeRecorder


@dataclass
class EvalArgs(RunArgs):
    render: bool = True
    best: bool = True
    n_envs: Optional[int] = 1
    n_episodes: int = 3
    deterministic_eval: Optional[bool] = None
    no_print_returns: bool = False
    wandb_run_path: Optional[str] = None
    video_path: Optional[str] = None
    override_hparams: Optional[Dict[str, Any]] = None
    visualize_model_path: Optional[str] = None
    thop: bool = False
    tensorboard_folder: Optional[str] = None


class Evaluation(NamedTuple):
    policy: Policy
    stats: EpisodesStats
    config: Config
    data_store_accessor: InProcessDataStoreAccessor


def evaluate_model(args: EvalArgs, root_dir: str) -> Evaluation:
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
    if args.video_path:
        env = VecEpisodeRecorder(
            env, args.video_path, max_video_length=18000, num_episodes=args.n_episodes
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
    if args.visualize_model_path or args.thop or args.tensorboard_folder:
        obs, _ = env.reset()
        get_action_mask = getattr(env, "get_action_mask", None)
        action_masks = batch_dict_keys(get_action_mask()) if get_action_mask else None
        act = policy.act(
            obs,
            deterministic=deterministic,
            action_masks=action_masks,
        )
        assert isinstance(obs, np.ndarray)
        t_obs = flatten_to_tensor(obs, device)
        t_act = flatten_actions_to_tensor(act, device)
        t_action_mask = (
            flatten_actions_to_tensor(action_masks, device)
            if action_masks is not None
            else None
        )
        inputs = (t_obs, t_act, t_action_mask)

        if args.tensorboard_folder:
            writer = SummaryWriter(args.tensorboard_folder)
            writer.add_graph(policy, inputs)
            writer.close()
        if args.visualize_model_path:
            import torchviz

            logp_a, _, _ = policy(*inputs)
            torchviz.make_dot(
                logp_a.mean(), params=dict(policy.named_parameters())
            ).render(args.visualize_model_path, format="png")
        if args.thop:
            import thop

            thop_out = thop.profile(policy, inputs=inputs)
            print(f"MACs: {thop_out[0] / 1e9:.2f}B. Params: {int(thop_out[1]):,}")

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
        data_store_accessor,
    )
