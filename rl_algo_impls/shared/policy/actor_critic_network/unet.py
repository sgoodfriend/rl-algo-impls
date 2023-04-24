from typing import Optional, Sequence, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import MultiDiscrete, Space

from rl_algo_impls.shared.actor import pi_forward
from rl_algo_impls.shared.actor.gridnet import GridnetDistribution
from rl_algo_impls.shared.actor.gridnet_decoder import Transpose
from rl_algo_impls.shared.module.utils import layer_init
from rl_algo_impls.shared.policy.actor_critic_network.network import (
    ACNForward,
    ActorCriticNetwork,
    default_hidden_sizes,
)
from rl_algo_impls.shared.policy.critic import CriticHead
from rl_algo_impls.shared.policy.policy import ACTIVATION


class UNetActorCriticNetwork(ActorCriticNetwork):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_plane_space: Space,
        v_hidden_sizes: Optional[Sequence[int]] = None,
        init_layers_orthogonal: bool = True,
        activation_fn: str = "tanh",
        cnn_layers_init_orthogonal: Optional[bool] = None,
    ) -> None:
        if cnn_layers_init_orthogonal is None:
            cnn_layers_init_orthogonal = True
        super().__init__()
        assert isinstance(action_space, MultiDiscrete)
        assert isinstance(action_plane_space, MultiDiscrete)
        self.range_size = np.max(observation_space.high) - np.min(observation_space.low)  # type: ignore
        self.map_size = len(action_space.nvec) // len(action_plane_space.nvec)  # type: ignore
        self.action_vec = action_plane_space.nvec  # type: ignore

        activation = ACTIVATION[activation_fn]

        def conv_relu(
            in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1
        ) -> nn.Module:
            return nn.Sequential(
                layer_init(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    cnn_layers_init_orthogonal,
                ),
                activation(),
            )

        def up_conv_relu(in_channels: int, out_channels: int) -> nn.Module:
            return nn.Sequential(
                layer_init(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    cnn_layers_init_orthogonal,
                ),
                activation(),
            )

        in_channels = observation_space.shape[0]  # type: ignore
        self.enc1 = conv_relu(in_channels, 32)
        self.enc2 = nn.Sequential(max_pool(), conv_relu(32, 64))
        self.enc3 = nn.Sequential(max_pool(), conv_relu(64, 128))
        self.enc4 = nn.Sequential(max_pool(), conv_relu(128, 256))
        self.enc5 = nn.Sequential(
            max_pool(), conv_relu(256, 512, kernel_size=1, padding=0)
        )

        self.dec4 = up_conv_relu(512, 256)
        self.dec3 = nn.Sequential(conv_relu(512, 256), up_conv_relu(256, 128))
        self.dec2 = nn.Sequential(conv_relu(256, 128), up_conv_relu(128, 64))
        self.dec1 = nn.Sequential(conv_relu(128, 64), up_conv_relu(64, 32))
        self.out = nn.Sequential(
            conv_relu(64, 32),
            layer_init(
                nn.Conv2d(32, self.action_vec.sum(), kernel_size=1, padding=0),
                cnn_layers_init_orthogonal,
                std=0.01,
            ),
            Transpose((0, 2, 3, 1)),
        )

        with torch.no_grad():
            cnn_out = torch.flatten(
                F.adaptive_avg_pool2d(
                    self.enc5(
                        self.enc4(
                            self.enc3(
                                self.enc2(
                                    self.enc1(
                                        self._preprocess(
                                            torch.as_tensor(observation_space.sample())
                                        )
                                    )
                                )
                            )
                        )
                    ),
                    output_size=1,
                ),
                start_dim=1,
            )

        v_hidden_sizes = (
            v_hidden_sizes
            if v_hidden_sizes is not None
            else default_hidden_sizes(observation_space)
        )
        self.critic_head = CriticHead(
            in_dim=cnn_out.shape[1:],
            hidden_sizes=v_hidden_sizes,
            activation=activation,
            init_layers_orthogonal=init_layers_orthogonal,
        )

    def _preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        return obs.float() / self.range_size

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
        assert (
            action_masks is not None
        ), f"No mask case unhandled in {self.__class__.__name__}"

        obs = self._preprocess(obs)
        e1 = self.enc1(obs)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        v = self.critic_head(F.adaptive_avg_pool2d(e5, output_size=1))

        d4 = self.dec4(e5)
        d3 = self.dec3(torch.cat((d4, e4), dim=1))
        d2 = self.dec2(torch.cat((d3, e3), dim=1))
        d1 = self.dec1(torch.cat((d2, e2), dim=1))
        logits = self.out(torch.cat((d1, e1), dim=1))

        pi = GridnetDistribution(
            int(np.prod(obs.shape[-2:])), self.action_vec, logits, action_masks
        )

        return ACNForward(pi_forward(pi, action), v)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self._preprocess(obs)
        e1 = self.enc1(obs)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        return self.critic_head(F.adaptive_avg_pool2d(e5, output_size=1))

    def reset_noise(self, batch_size: Optional[int] = None) -> None:
        pass

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return (self.map_size, len(self.action_vec))


def max_pool() -> nn.MaxPool2d:
    return nn.MaxPool2d(3, stride=2, padding=1)
