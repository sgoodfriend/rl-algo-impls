import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from rl_algo_impls.rollout.rollout import flatten_actions_to_tensor, flatten_to_tensor
from rl_algo_impls.runner.env_hyperparams import EnvHyperparams
from rl_algo_impls.runner.config import Config, RunArgs
from rl_algo_impls.shared.vec_env.device import get_device
from rl_algo_impls.runner.running_utils import (
    load_hyperparams,
    make_eval_policy,
    set_device_optimizations,
    set_seeds,
)
from rl_algo_impls.runner.wandb_load import load_player
from rl_algo_impls.shared.data_store.data_store_view import EvalDataStoreView
from rl_algo_impls.shared.data_store.in_process_data_store_accessor import (
    InProcessDataStoreAccessor,
)
from rl_algo_impls.shared.tensor_utils import batch_dict_keys
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces
from rl_algo_impls.shared.vec_env.make_env import make_eval_env


@dataclass
class ExportArgs(RunArgs):
    best: bool = True
    n_envs: Optional[int] = 1
    deterministic_eval: Optional[bool] = None
    wandb_run_path: Optional[str] = None
    override_hparams: Optional[Dict[str, Any]] = None
    thop: bool = True
    save_path: Optional[str] = None


def onnx_export(args: ExportArgs, root_dir: str):
    save_path = args.save_path
    assert save_path, f"save_path required"
    if args.wandb_run_path:
        import wandb

        api = wandb.Api()

        args, config, model_path = load_player(
            api, args.wandb_run_path, args, root_dir, args.best
        )
    else:
        hyperparams = load_hyperparams(args.algo, args.env)

        config = Config(args, hyperparams, root_dir)
        model_path = config.model_dir_path(best=args.best)

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
        render=False,
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
    torch.onnx.export(
        policy,
        inputs,
        save_path,
        verbose=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        input_names=("Obs", "Action", "Action Mask"),
        output_names=("logp_action", "entropy", "value"),
    )
    if args.thop:
        import thop

        thop_out = thop.profile(policy, inputs=inputs)
        print(f"MACs: {thop_out[0] / 1e9:.2f}B. Params: {int(thop_out[1]):,}")


if __name__ == "__main__":
    onnx_export(
        ExportArgs(
            algo="ppo",
            env="",
            wandb_run_path="sgoodfriend/rl-algo-impls-benchmarks/1ilo9yae",
            override_hparams={
                "map_paths": ["maps/16x16/basesWorkers16x16A.xml"],
            },
            save_path=os.path.expanduser("~/Desktop/model.onnx"),
        ),
        os.getcwd(),
    )
