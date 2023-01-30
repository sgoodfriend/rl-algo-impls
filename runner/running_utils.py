import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.backends.cudnn
import yaml

from gym.spaces import Box, Discrete
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Dict, Optional, Type, Union

from runner.names import Hyperparams
from shared.algorithm import Algorithm
from shared.callbacks.eval_callback import EvalCallback
from shared.policy.policy import Policy

from dqn.dqn import DQN
from dqn.policy import DQNPolicy
from vpg.vpg import VanillaPolicyGradient
from vpg.policy import VPGActorCritic
from ppo.ppo import PPO
from ppo.policy import PPOActorCritic

ALGOS: Dict[str, Type[Algorithm]] = {
    "dqn": DQN,
    "vpg": VanillaPolicyGradient,
    "ppo": PPO,
}

POLICIES: Dict[str, Type[Policy]] = {
    "dqn": DQNPolicy,
    "vpg": VPGActorCritic,
    "ppo": PPOActorCritic,
}

HYPERPARAMS_PATH = "hyperparams"


def base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        default="dqn",
        type=str,
        choices=list(ALGOS.keys()),
        nargs="+",
        help="Abbreviation(s) of algorithm(s)",
    )
    parser.add_argument(
        "--env",
        default="CartPole-v1",
        type=str,
        nargs="+",
        help="Name of environment(s) in gym",
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        nargs="*",
        help="Seeds to run experiment. Unset will do one run with no set seed",
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        default=True,
        type=bool,
        help="If seed set, set torch.use_deterministic_algorithms",
    )
    return parser


def load_hyperparams(algo: str, env_id: str, root_path: str) -> Hyperparams:
    hyperparams_path = os.path.join(root_path, HYPERPARAMS_PATH, f"{algo}.yml")
    with open(hyperparams_path, "r") as f:
        hyperparams_dict = yaml.safe_load(f)
    spec = gym.spec(env_id)
    if env_id in hyperparams_dict:
        return hyperparams_dict[env_id]
    elif "AtariEnv" in str(spec.entry_point) and "atari" in hyperparams_dict:
        return hyperparams_dict["atari"]
    else:
        raise ValueError(f"{env_id} not specified in {algo} hyperparameters file")


def get_device(device: str, env: VecEnv) -> torch.device:
    # cuda by default
    if device == "auto":
        device = "cuda"
    # Apple MPS is a second choice (sometimes) 
    if device == "cuda" and not torch.cuda.is_available():
        device = "mps"
    # If no MPS, fallback to cpu
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    # Simple environments like Discreet and 1-D Boxes might also be better
    # served with the CPU.
    if device == "mps":
        obs_space = env.observation_space
        if isinstance(obs_space, Discrete):
            device = "cpu"
        elif isinstance(obs_space, Box) and len(obs_space.shape) == 1:
            device = "cpu"
    print(f"Device: {device}")
    return torch.device(device)


def set_seeds(seed: Optional[int], use_deterministic_algorithms: bool) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(use_deterministic_algorithms)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def make_policy(
    algo: str,
    env: VecEnv,
    device: torch.device,
    load_path: Optional[str] = None,
    **kwargs,
) -> Policy:
    policy = POLICIES[algo](env, **kwargs).to(device)
    if load_path:
        policy.load(load_path)
    return policy


def plot_eval_callback(callback: EvalCallback, tb_writer: SummaryWriter, run_name: str):
    figure = plt.figure()
    cumulative_steps = [
        (idx + 1) * callback.step_freq for idx in range(len(callback.stats))
    ]
    plt.plot(
        cumulative_steps,
        [s.score.mean for s in callback.stats],
        "b-",
        label="mean",
    )
    plt.plot(
        cumulative_steps,
        [s.score.mean - s.score.std for s in callback.stats],
        "g--",
        label="mean-std",
    )
    plt.fill_between(
        cumulative_steps,
        [s.score.min for s in callback.stats],  # type: ignore
        [s.score.max for s in callback.stats],  # type: ignore
        facecolor="cyan",
        label="range",
    )
    plt.xlabel("Steps")
    plt.ylabel("Score")
    plt.legend()
    plt.title(f"Eval {run_name}")
    tb_writer.add_figure("eval", figure)


Scalar = Union[bool, str, float, int, None]


def flatten_hyperparameters(
    hyperparams: Hyperparams, args: Dict[str, Scalar]
) -> Dict[str, Scalar]:
    flattened = args.copy()
    for k, v in hyperparams.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                key = f"{k}/{sk}"
                if isinstance(sv, dict) or isinstance(sv, list):
                    flattened[key] = str(sv)
                else:
                    flattened[key] = sv
        else:
            flattened[k] = v
    return flattened
