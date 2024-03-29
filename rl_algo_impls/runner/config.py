import dataclasses
import inspect
import itertools
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union

from rl_algo_impls.runner.worker_hyperparams import WorkerHyperparams

RunArgsSelf = TypeVar("RunArgsSelf", bound="RunArgs")


@dataclass
class RunArgs:
    algo: str
    env: str
    seed: Optional[int] = None
    device_indexes: Optional[List[int]] = None

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
class TrainArgs(RunArgs):
    wandb_project_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: Sequence[str] = dataclasses.field(default_factory=list)
    wandb_group: Optional[str] = None


HyperparamsSelf = TypeVar("HyperparamsSelf", bound="Hyperparams")


@dataclass
class Hyperparams:
    process_mode: str = "sync"
    device: str = "auto"
    n_timesteps: Union[int, float] = 100_000
    env_hyperparams: Dict[str, Any] = dataclasses.field(default_factory=dict)
    policy_hyperparams: Dict[str, Any] = dataclasses.field(default_factory=dict)
    algo_hyperparams: Dict[str, Any] = dataclasses.field(default_factory=dict)
    eval_hyperparams: Dict[str, Any] = dataclasses.field(default_factory=dict)
    env_id: Optional[str] = None
    additional_keys_to_log: List[str] = dataclasses.field(default_factory=list)
    hyperparam_transitions_kwargs: Dict[str, Any] = dataclasses.field(
        default_factory=dict
    )
    rollout_hyperparams: Dict[str, Any] = dataclasses.field(default_factory=dict)
    rollout_type: Optional[str] = None
    device_hyperparams: Dict[str, Any] = dataclasses.field(default_factory=dict)
    lr_by_kl_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    checkpoints_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    worker_hyperparams: Dict[str, Any] = dataclasses.field(default_factory=dict)
    evaluate_after_training: bool = True

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
    gpu_ids: Optional[List[int]] = None
    run_id: str = datetime.now().isoformat()

    def __post_init__(self) -> None:
        self._worker_hyperparams = WorkerHyperparams(
            **self.hyperparams.worker_hyperparams
        )

    def seed(self, training: bool = True) -> Optional[int]:
        seed = self.args.seed
        if training or seed is None:
            return seed
        return seed + self.env_hyperparams.get("n_envs", 1)

    @property
    def process_mode(self) -> str:
        return self.hyperparams.process_mode

    @property
    def device(self) -> str:
        return self.hyperparams.device

    @property
    def n_timesteps(self) -> int:
        return int(self.hyperparams.n_timesteps)

    @property
    def evaluate_after_training(self) -> bool:
        return self.hyperparams.evaluate_after_training

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

    @property
    def rollout_hyperparams(self) -> Dict[str, Any]:
        return self.hyperparams.rollout_hyperparams

    @property
    def rollout_type(self) -> Optional[str]:
        return self.hyperparams.rollout_type

    @property
    def device_hyperparams(self) -> Dict[str, Any]:
        return self.hyperparams.device_hyperparams

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

    @property
    def worker_hyperparams(self) -> WorkerHyperparams:
        return self._worker_hyperparams

    def rollout_cuda_index(self, rollout_worker_idx: int) -> Optional[int]:
        if self.gpu_ids:
            gpu_idx = min(
                self._worker_hyperparams.rollout_gpu_index(rollout_worker_idx),
                len(self.gpu_ids) - 1,
            )
            return self.gpu_ids[gpu_idx]
        return None

    @property
    def evaluator_cuda_index(self) -> Optional[int]:
        if self.gpu_ids:
            gpu_idx = min(
                self._worker_hyperparams.evaluator_gpu_index, len(self.gpu_ids) - 1
            )
            return self.gpu_ids[gpu_idx]
        return None

    @property
    def inference_cuda_indexes(self) -> Optional[List[int]]:
        if self.gpu_ids:
            if not self._worker_hyperparams.inference_gpu_indexes:
                return [self.gpu_ids[0]]
            return [
                self.gpu_ids[gpu_idx % len(self.gpu_ids)]
                for gpu_idx in self._worker_hyperparams.inference_gpu_indexes
            ]
        return None

    @property
    def learner_cuda_index(self) -> Optional[int]:
        return self.gpu_ids[0] if self.gpu_ids else None

    @property
    def checkpoint_history_size(self) -> int:
        return self.hyperparams.checkpoints_kwargs.get("history_size", 0)
