import multiprocessing
import os
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Type, TypeVar

if TYPE_CHECKING:
    from rl_algo_impls.runner.config import Config

_MAX_NUM_THREADS = 32


def get_max_num_threads() -> int:
    cpu_count = multiprocessing.cpu_count()
    if cpu_count > _MAX_NUM_THREADS:
        return _MAX_NUM_THREADS
    elif cpu_count >= _MAX_NUM_THREADS // 2:
        return _MAX_NUM_THREADS // 2
    return cpu_count


def maybe_init_ray(config: "Config") -> None:
    os.environ["OMP_NUM_THREADS"] = str(get_max_num_threads())

    if config.process_mode != "async":
        return
    import ray

    worker_hyperparams = config.worker_hyperparams

    ray.init(
        namespace="rl_algo_impls",
        num_cpus=(
            worker_hyperparams.n_rollout_workers
            + worker_hyperparams.n_inference_workers
            + 1  # 1 for PolicyWorkerPool
            + 1  # RolloutGeneratorPool
            + 3  # 3 for data store, evaluator, and summary writer
        ),
    )


def init_ray_actor(
    num_threads: Optional[int] = None, cuda_visible_devices: Optional[List[int]] = None
):
    if num_threads is None:
        num_threads = get_max_num_threads()
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_visible_devices))
    import torch

    if cuda_visible_devices is not None:
        assert torch.cuda.device_count() == len(
            cuda_visible_devices
        ), f"GPU count mismatch: {torch.cuda.device_count()} vs expected {len(cuda_visible_devices)}"

    if torch.get_num_threads() != num_threads:
        torch.set_num_threads(num_threads)


EnvDataSelf = TypeVar("EnvDataSelf", bound="EnvData")


class EnvData(NamedTuple):
    env: Dict[str, str]
    torch: Dict[str, Any]
    multiprocessing: Dict[str, Any]
    os: Dict[str, Any]
    jax: Dict[str, Any]

    @classmethod
    def create(cls: Type[EnvDataSelf]) -> EnvDataSelf:
        try:
            import jax

            jax_data = {
                "local_device_count": jax.local_device_count(),
                "default_backend": jax.default_backend(),
            }
        except ImportError:
            jax_data = {}

        import torch

        return cls(
            env=dict(os.environ),
            torch={
                "cuda_available": torch.cuda.is_available(),
                "mps_available": torch.backends.mps.is_available(),
                "num_threads": torch.get_num_threads(),
            },
            multiprocessing={
                "cpu_count": multiprocessing.cpu_count(),
            },
            os={
                "cpu_count": os.cpu_count(),
            },
            jax=jax_data,
        )

    def diff(self: EnvDataSelf, other: EnvDataSelf) -> Dict[str, Any]:
        d = {}
        for field, _dict in self._asdict().items():
            other_dict = getattr(other, field)
            diff = dict_diff(_dict, other_dict)
            if diff:
                d[field] = diff
        return d


def dict_diff(lhs: Dict, rhs: Dict) -> Dict[str, Any]:
    diff = {}
    for k, v in lhs.items():
        if k not in rhs:
            if "missing" not in diff:
                diff["missing"] = []
            diff["missing"].append((k, v))
        elif v != rhs[k]:
            if "different" not in diff:
                diff["different"] = {}
            diff["different"][k] = (v, rhs[k])
    for k, v in rhs.items():
        if k not in lhs:
            if "extra" not in diff:
                diff["extra"] = []
            diff["extra"].append((k, v))
    return diff
