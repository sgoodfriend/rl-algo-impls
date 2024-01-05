import json
import multiprocessing
import os
from typing import Any, Dict, NamedTuple, Type, TypeVar

import jax

from rl_algo_impls.utils.ray import init_ray_actor

# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Don't overwrite CUDA_VISIBLE_DEVICES on ray workers (https://discuss.ray.io/t/how-to-stop-ray-from-managing-cuda-visible-devices/8767/2)
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

import ray
import torch


def ray_init_hook():
    os.environ.pop("OMP_NUM_THREADS", None)


ray.init()
# ray.init(runtime_env=RuntimeEnv(worker_process_setup_hook=ray_init_hook))
# ray.init(runtime_env=RuntimeEnv(env_vars={"OMP_NUM_THREADS": ""}))

EnvDataSelf = TypeVar("EnvDataSelf", bound="EnvData")


class EnvData(NamedTuple):
    env: Dict[str, str]
    torch: Dict[str, Any]
    multiprocessing: Dict[str, Any]
    os: Dict[str, Any]
    jax: Dict[str, Any]

    @classmethod
    def create(cls: Type[EnvDataSelf]) -> EnvDataSelf:
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
            jax={
                "local_device_count": jax.local_device_count(),
            },
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


# @ray.remote(num_cpus=2, runtime_env=RuntimeEnv(env_vars={"OMP_NUM_THREADS": ""}))
@ray.remote(num_cpus=2)
def get_ray_env_data() -> EnvData:
    init_ray_actor()
    return EnvData.create()


main_env_data = EnvData.create()
ray_env_data = ray.get(get_ray_env_data.remote())
print(json.dumps(main_env_data.diff(ray_env_data), indent=4, sort_keys=True))
