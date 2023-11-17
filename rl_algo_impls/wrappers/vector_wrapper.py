from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from gymnasium.core import ObsType
from gymnasium.experimental.vector.vector_env import ArrayType, VectorEnv, VectorWrapper

VecEnvStepReturn = Tuple[ObsType, ArrayType, ArrayType, ArrayType, dict]
VecEnvResetReturn = Tuple[ObsType, dict]
VecEnvMaskedResetReturn = Tuple[ObsType, np.ndarray, dict]


W = TypeVar("W", bound=VectorWrapper)


def find_wrapper(env: VectorEnv, wrapper_class: Type[W]) -> Optional[W]:
    current = env
    while current and current != current.unwrapped:
        if isinstance(current, wrapper_class):
            return current
        current = getattr(current, "env")
    return None


def get_info(infos: dict, key: Any, env_idx: int) -> Any:
    if key in infos:
        if isinstance(infos[key], dict):
            return _get_dict_idx(infos[key], env_idx)
        return infos[key][env_idx]
    return None


def _get_dict_idx(_dict: Dict[str, Union[Dict, np.ndarray]], env_idx: int) -> Any:
    return {
        k: (_get_dict_idx(v, env_idx) if isinstance(v, dict) else v[env_idx])
        for k, v in _dict.items()
    }


def get_infos(infos: dict, key: Any, num_envs: int, default_value: Any) -> List[Any]:
    if key in infos:
        assert len(infos[f"_{key}"]) == num_envs
        return [
            get_info(infos, key, idx) if is_set else default_value
            for idx, is_set in enumerate(infos[f"_{key}"])
        ]
    return [default_value for _ in range(num_envs)]


def filter_info(infos: dict, mask: Iterable[bool]) -> dict:
    _infos = {}
    for key, values in infos.items():
        if key.startswith("_"):
            continue
        if isinstance(values, dict):
            _infos[key] = _filter_dict(values, mask)
        else:
            _infos[key] = np.array(
                [v for v, m in zip(values, mask) if m], dtype=values.dtype
            )
        set_key = f"_{key}"
        _infos[set_key] = np.array(
            [s for s, m in zip(infos[set_key], mask) if m], dtype=np.bool_
        )
    return _infos


def _filter_dict(
    _dict: Dict[str, Union[Dict, np.ndarray]], mask: Iterable[bool]
) -> Dict[str, Union[Dict, np.ndarray]]:
    return {
        k: (_filter_dict(v, mask) if isinstance(v, dict) else v[mask])
        for k, v in _dict.items()
    }


def merge_info(env: VectorEnv, infos: Iterable[dict]) -> dict:
    _infos = {}
    for idx, info in enumerate(infos):
        env._add_info(_infos, info, idx)
    return _infos


def merge_infos(
    env: VectorEnv, infos: Iterable[dict], num_envs_per_info: Union[Sequence[int], int]
) -> dict:
    if isinstance(num_envs_per_info, int):
        num_envs_per_info = [num_envs_per_info for _ in infos]
    _infos = {}
    idx = 0
    for info, num_envs in zip(infos, num_envs_per_info):
        for k in info:
            if k.startswith("_"):
                continue
            if isinstance(info[k], dict):
                if k not in _infos:
                    info_array = {
                        kk: np.zeros(num_envs, dtype=vv.dtype)
                        for kk, vv in info[k].items()
                    }
                    array_mask = np.zeros(num_envs, dtype=np.bool_)
                else:
                    info_array, array_mask = _infos[k], _infos[f"_{k}"]
                for kk, vv in info[k].items():
                    info_array[kk][idx : idx + num_envs] = vv
            else:
                if k not in _infos:
                    info_array, array_mask = env._init_info_arrays(type(info[k].dtype))
                else:
                    info_array, array_mask = _infos[k], _infos[f"_{k}"]
                info_array[idx : idx + num_envs] = info[k]
            array_mask[idx : idx + num_envs] = info[f"_{k}"]
            _infos[k] = info_array
            _infos[f"_{k}"] = array_mask
        idx += num_envs
    return _infos


def extract_info(infos: dict, env_idx: int) -> Any:
    info = {}
    for key, values in infos.items():
        if key.startswith("_"):
            continue
        if isinstance(values, dict):
            info[key] = _extract_dict(values, env_idx)
        else:
            info[key] = values[env_idx]
    return info


def _extract_dict(_dict: Dict[str, Union[Dict, np.ndarray]], env_idx: int) -> Any:
    return {
        k: (_extract_dict(v, env_idx) if isinstance(v, dict) else v[env_idx])
        for k, v in _dict.items()
    }
