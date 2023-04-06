from typing import Optional, Tuple, Type

import gym
import torch.nn as nn
from gym.spaces import Box, Discrete, MultiDiscrete

from rl_algo_impls.shared.actor.actor import Actor
from rl_algo_impls.shared.actor.categorical import CategoricalActorHead
from rl_algo_impls.shared.actor.gaussian import GaussianActorHead
from rl_algo_impls.shared.actor.gridnet import GridnetActorHead
from rl_algo_impls.shared.actor.gridnet_decoder import GridnetDecoder
from rl_algo_impls.shared.actor.multi_discrete import MultiDiscreteActorHead
from rl_algo_impls.shared.actor.state_dependent_noise import (
    StateDependentNoiseActorHead,
)
from rl_algo_impls.shared.encoder import EncoderOutDim


def actor_head(
    action_space: gym.Space,
    in_dim: EncoderOutDim,
    hidden_sizes: Tuple[int, ...],
    init_layers_orthogonal: bool,
    activation: Type[nn.Module],
    log_std_init: float = -0.5,
    use_sde: bool = False,
    full_std: bool = True,
    squash_output: bool = False,
    actor_head_style: str = "single",
    action_plane_space: Optional[bool] = None,
) -> Actor:
    assert not use_sde or isinstance(
        action_space, Box
    ), "use_sde only valid if Box action_space"
    assert not squash_output or use_sde, "squash_output only valid if use_sde"
    if isinstance(action_space, Discrete):
        assert isinstance(in_dim, int)
        return CategoricalActorHead(
            action_space.n,  # type: ignore
            in_dim=in_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            init_layers_orthogonal=init_layers_orthogonal,
        )
    elif isinstance(action_space, Box):
        assert isinstance(in_dim, int)
        if use_sde:
            return StateDependentNoiseActorHead(
                action_space.shape[0],  # type: ignore
                in_dim=in_dim,
                hidden_sizes=hidden_sizes,
                activation=activation,
                init_layers_orthogonal=init_layers_orthogonal,
                log_std_init=log_std_init,
                full_std=full_std,
                squash_output=squash_output,
            )
        else:
            return GaussianActorHead(
                action_space.shape[0],  # type: ignore
                in_dim=in_dim,
                hidden_sizes=hidden_sizes,
                activation=activation,
                init_layers_orthogonal=init_layers_orthogonal,
                log_std_init=log_std_init,
            )
    elif isinstance(action_space, MultiDiscrete):
        if actor_head_style == "single":
            return MultiDiscreteActorHead(
                action_space.nvec,  # type: ignore
                in_dim=in_dim,
                hidden_sizes=hidden_sizes,
                activation=activation,
                init_layers_orthogonal=init_layers_orthogonal,
            )
        elif actor_head_style == "gridnet":
            assert isinstance(action_plane_space, MultiDiscrete)
            return GridnetActorHead(
                len(action_space.nvec) // len(action_plane_space.nvec),  # type: ignore
                action_plane_space.nvec,  # type: ignore
                in_dim=in_dim,
                hidden_sizes=hidden_sizes,
                activation=activation,
                init_layers_orthogonal=init_layers_orthogonal,
            )
        elif actor_head_style == "gridnet_decoder":
            assert isinstance(action_plane_space, MultiDiscrete)
            return GridnetDecoder(
                len(action_space.nvec) // len(action_plane_space.nvec),  # type: ignore
                action_plane_space.nvec,  # type: ignore
                in_dim=in_dim,
                activation=activation,
                init_layers_orthogonal=init_layers_orthogonal,
            )
        else:
            raise ValueError(f"Doesn't support actor_head_style {actor_head_style}")
    else:
        raise ValueError(f"Unsupported action space: {action_space}")
