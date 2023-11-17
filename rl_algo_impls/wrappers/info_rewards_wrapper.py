import collections.abc
from typing import Dict, List, Optional, Union

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

    def step(self, action) -> VecEnvStepReturn:
        o, r, terminations, truncations, infos = super().step(action)
        r_to_add = np.stack(
            [get_by_path(infos, i_path) for i_path in self.info_paths], axis=-1
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


def get_by_path(
    info: Dict[str, Union[Dict, np.ndarray]], path: List[str]
) -> np.ndarray:
    if len(path) == 0:
        return info
    return get_by_path(info[path[0]], path[1:])
