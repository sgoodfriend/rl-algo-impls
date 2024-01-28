import collections.abc
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from rl_algo_impls.wrappers.vector_wrapper import (
    VecEnvStepReturn,
    VectorEnv,
    VectorWrapper,
)


class InfoRewardsWrapper(VectorWrapper):
    def __init__(
        self,
        env: VectorEnv,
        info_paths: List[List[str]],
        episode_end: Union[bool, List[bool]] = True,
        multiplier: Union[None, float, List[float]] = None,
        flatten_info: bool = False,
    ):
        super().__init__(env)
        self.info_paths = info_paths
        if isinstance(episode_end, collections.abc.Sequence):
            self.episode_end = np.array(episode_end, dtype=np.bool_)
        else:
            self.episode_end = np.full(
                (len(self.info_paths),), episode_end, dtype=np.bool_
            )
        if isinstance(multiplier, collections.abc.Sequence):
            self.multiplier = np.array(multiplier, dtype=np.float32)
        elif multiplier is not None and multiplier != 1.0:
            self.multiplier = np.full(
                (len(self.info_paths),), multiplier, dtype=np.float32
            )
        else:
            self.multiplier = None
        self.flatten_info = flatten_info

    def step(self, action) -> VecEnvStepReturn:
        o, r, terminations, truncations, infos = super().step(action)
        r_to_add = np.stack(
            [
                get_by_path(infos, i_path, self.flatten_info)
                for i_path in self.info_paths
            ],
            axis=-1,
        )
        if self.episode_end.any():
            r_to_add = np.where(
                np.logical_or(
                    np.logical_or(terminations, truncations)[:, None],
                    ~self.episode_end[None, :],
                ),
                r_to_add,
                0,
            )
        if self.multiplier is not None:
            r_to_add *= self.multiplier[None, :]
        if len(r.shape) == 1:
            r = np.expand_dims(r, axis=-1)
        rewards = np.concatenate([r, r_to_add], axis=-1)
        return o, rewards, terminations, truncations, infos

    @property
    def reward_shape(self) -> Tuple[int, ...]:
        if hasattr(self.env, "reward_shape"):
            _reward_shape = getattr(self.env, "reward_shape")
            assert isinstance(_reward_shape, tuple)
        else:
            _reward_shape: Tuple[int, ...] = ()
        if len(_reward_shape):
            _reward_shape = (_reward_shape[0] + len(self.info_paths),)
        else:
            _reward_shape = (1 + len(self.info_paths),)
        return _reward_shape


def get_by_path(
    info: Dict[str, Union[Dict, np.ndarray]], path: List[str], flatten: bool = False
) -> np.ndarray:
    if len(path) == 0:
        return info
    _info = info[path[0]]
    if flatten:
        _info_dict: DefaultDict[str, List[float]] = collections.defaultdict(list)
        for ind_info in _info:
            for k, v in ind_info.items():
                _info_dict[k].append(v)
        _info = {k: np.array(v, dtype=np.float32) for k, v in _info_dict.items()}
    return get_by_path(_info, path[1:])
