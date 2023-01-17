import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.backends.cudnn
import yaml

from dataclasses import dataclass
from datetime import datetime
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Any, Callable, Dict, List, Optional, Type, TypedDict, Union

from shared.algorithm import Algorithm
from shared.callbacks.eval_callback import EvalCallback
from shared.policy import Policy
from shared.stats import EpisodesStats

from dqn.dqn import DQN
from dqn.policy import DQNPolicy
from vpg.vpg import VanillaPolicyGradient
from vpg.policy import ActorCritic

ALGOS: Dict[str, Type[Algorithm]] = {
    "dqn": DQN,
    "vpg": VanillaPolicyGradient,
}

POLICIES: Dict[str, Type[Policy]] = {
    "dqn": DQNPolicy,
    "vpg": ActorCritic,
}

HYPERPARAMS_PATH = "hyperparams"


class Hyperparams(TypedDict, total=False):
    device: str
    n_timesteps: Union[int, float]
    env_hyperparams: Dict
    policy_hyperparams: Dict
    algo_hyperparams: Dict


def load_hyperparams(algo: str, env_id: str, root_path: str) -> Hyperparams:
    hyperparams_path = os.path.join(root_path, HYPERPARAMS_PATH, f"{algo}.yml")
    with open(hyperparams_path, "r") as f:
        hyperparams_dict = yaml.safe_load(f)
    return hyperparams_dict[env_id]


def set_seeds(seed: Optional[int], use_deterministic_algorithms: bool) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(use_deterministic_algorithms)


def make_env(
    env_id: str,
    seed: Optional[int],
    render: bool = False,
    n_envs: int = 1,
    frame_stack: int = 1,
    make_kwargs: Optional[Dict[str, Any]] = None,
    no_reward_timeout_steps: Optional[int] = None,
    vec_env_class: str = "dummy",
) -> VecEnv:
    if "BulletEnv" in env_id:
        import pybullet_envs

    make_kwargs = make_kwargs if make_kwargs is not None else {}
    if "BulletEnv" in env_id and render:
        make_kwargs["render"] = True

    spec = gym.spec(env_id)

    def make(idx: int) -> Callable[[], gym.Env]:
        def _make() -> gym.Env:
            if "AtariEnv" in spec.entry_point:  # type: ignore
                from gym.wrappers.atari_preprocessing import AtariPreprocessing
                from gym.wrappers.frame_stack import FrameStack

                env = gym.make(env_id, **make_kwargs)
                env = AtariPreprocessing(env)
                env = FrameStack(env, frame_stack)
            elif "CarRacing" in env_id:
                from gym.wrappers.resize_observation import ResizeObservation
                from gym.wrappers.gray_scale_observation import GrayScaleObservation
                from gym.wrappers.frame_stack import FrameStack

                env = gym.make(env_id, verbose=0, **make_kwargs)
                env = ResizeObservation(env, (64, 64))
                env = GrayScaleObservation(env, keep_dim=False)
                env = FrameStack(env, frame_stack)
            else:
                env = gym.make(env_id, **make_kwargs)

            if no_reward_timeout_steps:
                from wrappers.no_reward_timeout import NoRewardTimeout

                env = NoRewardTimeout(env, no_reward_timeout_steps)

            if seed is not None:
                env.seed(seed + idx)
                env.action_space.seed(seed + idx)
                env.observation_space.seed(seed + idx)

            return env

        return _make

    VecEnvClass = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[vec_env_class]
    return VecEnvClass([make(i) for i in range(n_envs)])


def make_policy(
    algo: str,
    env: VecEnv,
    device: torch.device,
    load_path: Optional[str] = None,
    **kwargs,
) -> Policy:
    policy = POLICIES[algo](env, device, **kwargs)
    if load_path:
        policy.load(load_path)
    return policy


@dataclass
class Names:
    algo_name: str
    env_name: str
    hyperparams: Hyperparams
    root_dir: str
    run_id: str = datetime.now().isoformat()

    @property
    def model_name(self) -> str:
        model_name = f"{self.algo_name}-{self.env_name}"
        make_kwargs = self.hyperparams.get("env_hyperparams", {}).get("make_kwargs", {})
        if make_kwargs:
            for k, v in make_kwargs.items():
                if type(v) == bool and v:
                    model_name += f"-{k}"
                elif type(v) == int and v:
                    model_name += f"-{k}{v}"
                else:
                    model_name += f"-{v}"
        return model_name

    @property
    def run_name(self) -> str:
        return f"{self.model_name}-{self.run_id}"

    @property
    def saved_models_dir(self) -> str:
        return os.path.join(self.root_dir, "saved_models")

    def model_path(
        self,
        best: bool = False,
        include_run_id: bool = False,
    ) -> str:
        model_file_name = (
            (self.run_name if include_run_id else self.model_name)
            + ("-best" if best else "")
            + ".pt"
        )
        return os.path.join(self.saved_models_dir, model_file_name)

    @property
    def runs_dir(self) -> str:
        return os.path.join(self.root_dir, "runs")

    @property
    def tensorboard_summary_path(self) -> str:
        return os.path.join(self.runs_dir, self.run_name)

    @property
    def logs_path(self) -> str:
        return os.path.join(self.runs_dir, f"log.yml")


def plot_training(history: List[EpisodesStats], tb_writer: SummaryWriter):
    figure = plt.figure()
    cumulative_steps = []
    for es in history:
        cumulative_steps.append(
            es.length.sum() + (cumulative_steps[-1] if cumulative_steps else 0)
        )

    plt.plot(cumulative_steps, [es.score.mean for es in history])
    plt.fill_between(
        cumulative_steps,
        [es.score.min for es in history],  # type: ignore
        [es.score.max for es in history],  # type: ignore
        facecolor="cyan",
    )
    plt.xlabel("Steps")
    plt.ylabel("Score")
    tb_writer.add_figure("train", figure)


def plot_eval_callback(callback: EvalCallback, tb_writer: SummaryWriter):
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
