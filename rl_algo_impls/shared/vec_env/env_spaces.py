from typing import Dict, NamedTuple, Optional, Tuple, Type, TypeVar, Union

import gymnasium

from rl_algo_impls.shared.vec_env.action_shape import get_action_shape
from rl_algo_impls.wrappers.vector_wrapper import VectorEnv

EnvSpacesSelf = TypeVar("EnvSpacesSelf", bound="EnvSpaces")


class EnvSpaces(NamedTuple):
    single_observation_space: gymnasium.Space
    single_action_space: gymnasium.Space
    action_plane_space: Optional[gymnasium.Space]
    action_shape: Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]
    num_envs: int
    reward_shape: Tuple[int, ...]

    @classmethod
    def from_vec_env(cls: Type[EnvSpacesSelf], env: VectorEnv) -> EnvSpacesSelf:
        return cls(
            single_observation_space=env.single_observation_space,
            single_action_space=env.single_action_space,
            action_plane_space=getattr(env, "action_plane_space", None),
            action_shape=get_action_shape(env),
            num_envs=env.num_envs,
            reward_shape=getattr(env, "reward_shape", ()),
        )
