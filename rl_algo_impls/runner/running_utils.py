import argparse
import json
import logging
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

import gymnasium
import numpy as np
import torch
import torch.backends.cudnn
import yaml

import wandb
from rl_algo_impls.a2c.a2c import A2C
from rl_algo_impls.acbc.acbc import ACBC
from rl_algo_impls.ppo.appo import APPO
from rl_algo_impls.ppo.ppo import PPO
from rl_algo_impls.runner.config import Config, Hyperparams
from rl_algo_impls.runner.wandb_load import load_player
from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.data_store.data_store_data import LearnerInitializeData
from rl_algo_impls.shared.data_store.data_store_view import LearnerDataStoreView
from rl_algo_impls.shared.data_store.in_process_data_store_accessor import (
    InProcessDataStoreAccessor,
)
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces
from rl_algo_impls.shared.vec_env.utils import import_for_env_id

ALGOS: Dict[str, Type[Algorithm]] = {
    # "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
    "acbc": ACBC,
    "appo": APPO,
}
POLICIES: Dict[str, Type[Policy]] = {
    # "dqn": DQNPolicy,
    "ppo": ActorCritic,
    "a2c": ActorCritic,
    "acbc": ActorCritic,
    "appo": ActorCritic,
}
HYPERPARAMS_PATH = "hyperparams"


def base_parser(multiple: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        default=["dqn"],
        type=str,
        choices=list(ALGOS.keys()),
        nargs="+" if multiple else 1,
        help="Abbreviation(s) of algorithm(s)",
    )
    parser.add_argument(
        "--env",
        default=["CartPole-v1"],
        type=str,
        nargs="+" if multiple else 1,
        help="Name of environment(s) in gymnasium",
    )
    parser.add_argument(
        "--seed",
        default=[1],
        type=int,
        nargs="*" if multiple else "?",
        help="Seeds to run experiment. Unset will do one run with no set seed",
    )
    parser.add_argument(
        "--device-indexes",
        type=int,
        nargs="*" if multiple else "?",
        help="GPU device indexes to use. Unset will pick free GPU",
    )
    return parser


def load_hyperparams(algo: str, env_id: str) -> Hyperparams:
    return Hyperparams(**load_hyperparam_dict(algo, env_id))


def load_hyperparam_dict(algo: str, env_id: str) -> Dict[str, Any]:
    hp_dict = load_hyperparam_dict_by_env_id(
        algo, env_id
    ) or load_hyperparam_dict_by_algo(algo, env_id)
    if hp_dict is None:
        raise ValueError(f"({algo},{env_id}) hyperparameters not specified")
    return hp_dict


def load_hyperparam_dict_by_algo(algo: str, env_id: str) -> Optional[Dict[str, Any]]:
    root_path = Path(__file__).parent.parent
    hyperparams_path = os.path.join(root_path, HYPERPARAMS_PATH, f"{algo}.yml")
    with open(hyperparams_path, "r") as f:
        hyperparams_dict = yaml.safe_load(f)

    if env_id in hyperparams_dict:
        return hyperparams_dict[env_id]

    import_for_env_id(env_id)
    spec = gymnasium.spec(env_id)
    entry_point_name = str(spec.entry_point)  # type: ignore
    if "AtariEnv" in entry_point_name and "_atari" in hyperparams_dict:
        return hyperparams_dict["_atari"]
    elif "gym_microrts" in entry_point_name and "_microrts" in hyperparams_dict:
        return hyperparams_dict["_microrts"]

    return None


def load_hyperparam_dict_by_env_id(algo: str, env_id: str) -> Optional[Dict[str, Any]]:
    root_path = Path(__file__).parent.parent
    env_prefix = env_id.split("-")[0]
    hyperparams_path = os.path.join(
        root_path, HYPERPARAMS_PATH, f"{algo}-{env_prefix}.yml"
    )
    if not os.path.exists(hyperparams_path):
        return None
    with open(hyperparams_path, "r") as f:
        hyperparams_dict = yaml.safe_load(f)

    if env_id in hyperparams_dict:
        return hyperparams_dict[env_id]
    return None


def set_device_optimizations(
    device: torch.device,
    set_float32_matmul_precision: Optional[str] = None,
    use_deterministic_algorithms: bool = True,
) -> None:
    torch.use_deterministic_algorithms(use_deterministic_algorithms)
    if device.type == "cuda":
        if set_float32_matmul_precision:
            logging.info(
                f"Setting torch.set_float32_matmul_precision to {set_float32_matmul_precision}"
            )
            torch.set_float32_matmul_precision(set_float32_matmul_precision)


def set_seeds(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Stop warning and it would introduce stochasticity if I was using TF
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def make_eval_policy(
    config: Config,
    env_spaces: EnvSpaces,
    device: torch.device,
    data_store_accessor: InProcessDataStoreAccessor,
    load_path: Optional[str] = None,
    load_run_path: Optional[str] = None,
    load_run_path_best: bool = True,
    **kwargs,
) -> Policy:
    policy = POLICIES[config.algo](
        env_spaces,
        **kwargs,
    ).to(device)
    data_store_accessor._data_store.policy = policy
    if not load_path and load_run_path:
        api = wandb.Api()
        _, _, load_path = load_player(
            api, load_run_path, config.args, config.root_dir, load_run_path_best
        )
        assert load_path
    if load_path:
        data_store_accessor.load(load_path)
    return policy


def initialize_policy_algo_data_store_view(
    config: Config,
    env_spaces: EnvSpaces,
    device: torch.device,
    data_store_accessor: AbstractDataStoreAccessor,
    tb_writer: AbstractSummaryWrapper,
) -> Tuple[Policy, Algorithm, LearnerDataStoreView]:
    policy_kwargs = dict(config.policy_hyperparams)

    load_path = policy_kwargs.pop("load_path", None)
    load_run_path = policy_kwargs.pop("load_run_path", None)
    load_run_path_best = policy_kwargs.pop("load_run_path_best", True)
    if not load_path and load_run_path:
        api = wandb.Api()
        _, _, load_path = load_player(
            api, load_run_path, config.args, config.root_dir, load_run_path_best
        )
        assert load_path

    policy = POLICIES[config.algo](env_spaces, **policy_kwargs).to(device)

    num_parameters = policy.num_parameters()
    num_trainable_parameters = policy.num_trainable_parameters()
    tb_writer.update_summary(
        {
            "num_parameters": num_parameters,
            "num_trainable_parameters": num_trainable_parameters,
        }
    )

    algo = ALGOS[config.algo](
        policy,
        device,
        tb_writer,
        **config.algo_hyperparams,
    )
    data_store_accessor.initialize_learner(
        LearnerInitializeData(policy=policy, algo=algo, load_path=load_path)
    )
    return policy, algo, LearnerDataStoreView(data_store_accessor, device)


Scalar = Union[bool, str, float, int, None]


def hparam_dict(
    hyperparams: Hyperparams, args: Dict[str, Union[Scalar, list]]
) -> Dict[str, Scalar]:
    flattened = args.copy()
    for k, v in flattened.items():
        if isinstance(v, list):
            flattened[k] = json.dumps(v)
    for k, v in asdict(hyperparams).items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                key = f"{k}/{sk}"
                if isinstance(sv, dict) or isinstance(sv, list):
                    flattened[key] = str(sv)
                else:
                    flattened[key] = sv
        elif isinstance(v, list):
            flattened[k] = json.dumps(v)
        else:
            flattened[k] = v  # type: ignore
    return flattened  # type: ignore
