import dataclasses
import inspect
import itertools
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

RunArgsSelf = TypeVar("RunArgsSelf", bound="RunArgs")


@dataclass
class RunArgs:
    algo: str
    env: str
    seed: Optional[int] = None
    use_deterministic_algorithms: bool = True

    @classmethod
    def expand_from_dict(
        cls: Type[RunArgsSelf], d: Dict[str, Any]
    ) -> List[RunArgsSelf]:
        maybe_listify = lambda v: [v] if isinstance(v, str) or isinstance(v, int) else v
        algos = maybe_listify(d["algo"])
        envs = maybe_listify(d["env"])
        seeds = maybe_listify(d["seed"])
        args = []
        for algo, env, seed in itertools.product(algos, envs, seeds):
            _d = d.copy()
            _d.update({"algo": algo, "env": env, "seed": seed})
            args.append(cls(**_d))
        return args


@dataclass
class EnvHyperparams:
    env_type: str = "gymvec"
    n_envs: int = 1
    frame_stack: int = 1
    make_kwargs: Optional[Dict[str, Any]] = None
    no_reward_timeout_steps: Optional[int] = None
    no_reward_fire_steps: Optional[int] = None
    vec_env_class: str = "sync"
    normalize: bool = False
    normalize_kwargs: Optional[Dict[str, Any]] = None
    rolling_length: int = 100
    train_record_video: bool = False
    video_step_interval: Union[int, float] = 1_000_000
    initial_steps_to_truncate: Optional[int] = None
    clip_atari_rewards: bool = True
    normalize_type: Optional[str] = None
    mask_actions: bool = False
    bots: Optional[Dict[str, int]] = None
    self_play_kwargs: Optional[Dict[str, Any]] = None
    selfplay_bots: Optional[Dict[str, int]] = None


HyperparamsSelf = TypeVar("HyperparamsSelf", bound="Hyperparams")


@dataclass
class Hyperparams:
    device: str = "auto"
    n_timesteps: Union[int, float] = 100_000
    env_hyperparams: Dict[str, Any] = dataclasses.field(default_factory=dict)
    policy_hyperparams: Dict[str, Any] = dataclasses.field(default_factory=dict)
    algo_hyperparams: Dict[str, Any] = dataclasses.field(default_factory=dict)
    eval_hyperparams: Dict[str, Any] = dataclasses.field(default_factory=dict)
    env_id: Optional[str] = None
    additional_keys_to_log: List[str] = dataclasses.field(default_factory=list)
    reward_decay_callback: bool = False
    reward_decay_callback_kwargs: Dict[str, Any] = dataclasses.field(
        default_factory=dict
    )
    lux_hyperparam_transitions_kwargs: Dict[str, Any] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def from_dict_with_extra_fields(
        cls: Type[HyperparamsSelf], d: Dict[str, Any]
    ) -> HyperparamsSelf:
        return cls(
            **{k: v for k, v in d.items() if k in inspect.signature(cls).parameters}
        )


@dataclass
class Config:
    args: RunArgs
    hyperparams: Hyperparams
    root_dir: str
    run_id: str = datetime.now().isoformat()

    def seed(self, training: bool = True) -> Optional[int]:
        seed = self.args.seed
        if training or seed is None:
            return seed
        return seed + self.env_hyperparams.get("n_envs", 1)

    @property
    def device(self) -> str:
        return self.hyperparams.device

    @property
    def n_timesteps(self) -> int:
        return int(self.hyperparams.n_timesteps)

    @property
    def env_hyperparams(self) -> Dict[str, Any]:
        return self.hyperparams.env_hyperparams

    @property
    def policy_hyperparams(self) -> Dict[str, Any]:
        return self.hyperparams.policy_hyperparams

    @property
    def algo_hyperparams(self) -> Dict[str, Any]:
        return self.hyperparams.algo_hyperparams

    @property
    def eval_hyperparams(self) -> Dict[str, Any]:
        return self.hyperparams.eval_hyperparams

    def eval_callback_params(self) -> Dict[str, Any]:
        eval_hyperparams = self.eval_hyperparams.copy()
        if "env_overrides" in eval_hyperparams:
            del eval_hyperparams["env_overrides"]
        return eval_hyperparams

    @property
    def algo(self) -> str:
        return self.args.algo

    @property
    def env_id(self) -> str:
        return self.hyperparams.env_id or self.args.env

    @property
    def additional_keys_to_log(self) -> List[str]:
        return self.hyperparams.additional_keys_to_log

    def model_name(self, include_seed: bool = True) -> str:
        # Use arg env name instead of environment name
        parts = [self.algo, self.args.env]
        if include_seed and self.args.seed is not None:
            parts.append(f"S{self.args.seed}")

        # Assume that the custom arg name already has the necessary information
        if not self.hyperparams.env_id:
            make_kwargs = self.env_hyperparams.get("make_kwargs", {})
            if make_kwargs:
                for k, v in make_kwargs.items():
                    if type(v) == bool and v:
                        parts.append(k)
                    elif type(v) == int and v:
                        parts.append(f"{k}{v}")
                    else:
                        parts.append(str(v))

        return "-".join(parts)

    def run_name(self, include_seed: bool = True) -> str:
        parts = [self.model_name(include_seed=include_seed), self.run_id]
        return "-".join(parts)

    @property
    def saved_models_dir(self) -> str:
        return os.path.join(self.root_dir, "saved_models")

    @property
    def downloaded_models_dir(self) -> str:
        return os.path.join(self.root_dir, "downloaded_models")

    def model_dir_name(
        self,
        best: bool = False,
        extension: str = "",
    ) -> str:
        return self.model_name() + ("-best" if best else "") + extension

    def model_dir_path(self, best: bool = False, downloaded: bool = False) -> str:
        return os.path.join(
            self.saved_models_dir if not downloaded else self.downloaded_models_dir,
            self.model_dir_name(best=best),
        )

    @property
    def runs_dir(self) -> str:
        return os.path.join(self.root_dir, "runs")

    @property
    def tensorboard_summary_path(self) -> str:
        return os.path.join(self.runs_dir, self.run_name())

    @property
    def logs_path(self) -> str:
        return os.path.join(self.runs_dir, f"log.yml")

    @property
    def videos_dir(self) -> str:
        return os.path.join(self.root_dir, "videos")

    @property
    def video_prefix(self) -> str:
        return os.path.join(self.videos_dir, self.model_name())

    @property
    def videos_path(self) -> str:
        return os.path.join(self.videos_dir, self.model_name())
