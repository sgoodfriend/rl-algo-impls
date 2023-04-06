from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from gym.spaces import Space

from rl_algo_impls.shared.actor import actor_head
from rl_algo_impls.shared.encoder import Encoder
from rl_algo_impls.shared.policy.actor_critic_network.network import (
    ACNForward,
    ActorCriticNetwork,
    default_hidden_sizes,
)
from rl_algo_impls.shared.policy.critic import CriticHead
from rl_algo_impls.shared.policy.policy import ACTIVATION


class SeparateActorCriticNetwork(ActorCriticNetwork):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        pi_hidden_sizes: Optional[Sequence[int]] = None,
        v_hidden_sizes: Optional[Sequence[int]] = None,
        init_layers_orthogonal: bool = True,
        activation_fn: str = "tanh",
        log_std_init: float = -0.5,
        use_sde: bool = False,
        full_std: bool = True,
        squash_output: bool = False,
        cnn_flatten_dim: int = 512,
        cnn_style: str = "nature",
        cnn_layers_init_orthogonal: Optional[bool] = None,
        impala_channels: Sequence[int] = (16, 32, 32),
        actor_head_style: str = "single",
        action_plane_space: Optional[Space] = None,
    ) -> None:
        super().__init__()

        pi_hidden_sizes = (
            pi_hidden_sizes
            if pi_hidden_sizes is not None
            else default_hidden_sizes(observation_space)
        )
        v_hidden_sizes = (
            v_hidden_sizes
            if v_hidden_sizes is not None
            else default_hidden_sizes(observation_space)
        )

        activation = ACTIVATION[activation_fn]
        self._feature_extractor = Encoder(
            observation_space,
            activation,
            init_layers_orthogonal=init_layers_orthogonal,
            cnn_flatten_dim=cnn_flatten_dim,
            cnn_style=cnn_style,
            cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
            impala_channels=impala_channels,
        )
        self._pi = actor_head(
            action_space,
            self._feature_extractor.out_dim,
            tuple(pi_hidden_sizes),
            init_layers_orthogonal,
            activation,
            log_std_init=log_std_init,
            use_sde=use_sde,
            full_std=full_std,
            squash_output=squash_output,
            actor_head_style=actor_head_style,
            action_plane_space=action_plane_space,
        )

        v_encoder = Encoder(
            observation_space,
            activation,
            init_layers_orthogonal=init_layers_orthogonal,
            cnn_flatten_dim=cnn_flatten_dim,
            cnn_style=cnn_style,
            cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
        )
        self._v = nn.Sequential(
            v_encoder,
            CriticHead(
                in_dim=v_encoder.out_dim,
                hidden_sizes=v_hidden_sizes,
                activation=activation,
                init_layers_orthogonal=init_layers_orthogonal,
            ),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> ACNForward:
        return self._distribution_and_value(
            obs, action=action, action_masks=action_masks
        )

    def distribution_and_value(
        self, obs: torch.Tensor, action_masks: Optional[torch.Tensor] = None
    ) -> ACNForward:
        return self._distribution_and_value(obs, action_masks=action_masks)

    def _distribution_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
    ) -> ACNForward:
        pi_forward = self._pi(
            self._feature_extractor(obs), actions=action, action_masks=action_masks
        )
        v = self._v(obs)
        return ACNForward(pi_forward, v)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self._v(obs)

    def reset_noise(self, batch_size: int) -> None:
        self._pi.sample_weights(batch_size=batch_size)

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return self._pi.action_shape
