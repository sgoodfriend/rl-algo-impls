import multiprocessing
import os
from typing import Any, Dict, NamedTuple, Optional, Type, TypeVar

import jax
import torch


def init_ray_actor(num_threads: Optional[int] = None):
    if num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
    else:
        os.environ.pop("OMP_NUM_THREADS", None)
        num_threads = multiprocessing.cpu_count()
    import torch

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
                "default_backend": jax.default_backend(),
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
