from typing import Dict, Optional, Tuple, Union

import gymnasium
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import MultiDiscrete

from rl_algo_impls.wrappers.vector_env_render_compat import VectorEnv

ActionShape = Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]


def get_action_shape(vec_env: VectorEnv) -> ActionShape:
    action_space = vec_env.single_action_space
    action_plane_space = getattr(vec_env, "action_plane_space", None)
    if isinstance(action_space, DictSpace):
        assert action_plane_space is not None, f"action_plane_space is None"
        action_space_per_position = action_space["per_position"]
        assert isinstance(action_space_per_position, MultiDiscrete)
        action_space_pick_position = action_space["pick_position"]
        assert isinstance(action_space_pick_position, MultiDiscrete)
        return {
            "per_position": _get_multidiscrete_action_shape(
                action_space_per_position, action_plane_space
            ),
            "pick_position": _get_multidiscrete_action_shape(
                action_space_pick_position, None
            ),
        }
    elif isinstance(action_space, MultiDiscrete):
        return _get_multidiscrete_action_shape(action_space, action_plane_space)
    elif isinstance(action_space, gymnasium.spaces.Discrete):
        return ()
    elif isinstance(action_space, gymnasium.spaces.Box):
        return (action_space.shape[0],)
    else:
        raise ValueError(f"action_space {action_space} not supported")


def _get_multidiscrete_action_shape(
    action_space: MultiDiscrete, action_plane_space: Optional[MultiDiscrete]
) -> Tuple[int, ...]:
    if action_plane_space is None:
        return (len(action_space.nvec),)
    else:
        map_size = len(action_space.nvec) // len(action_plane_space.nvec)
        return (map_size, len(action_plane_space.nvec))
