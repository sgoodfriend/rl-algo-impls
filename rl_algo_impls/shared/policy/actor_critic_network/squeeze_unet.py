import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
from gym.spaces import Box, Space

from rl_algo_impls.shared.module.utils import layer_init
from rl_algo_impls.shared.policy.actor_critic_network.backbone_actor_critic import (
    BackboneActorCritic,
)
from rl_algo_impls.shared.policy.actor_critic_network.double_cone import ResidualBlock


class SqueezeUnetBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels_per_level: List[int],
        strides_per_level: List[Union[int, List[int]]],
        encoder_residual_blocks_per_level: List[int],
        decoder_residual_blocks_per_level: List[int],
        conv_strides_per_level: Optional[List[Union[int, List[int]]]] = None,
        init_layers_orthogonal: bool = False,
        increment_kernel_size_on_down_conv: bool = False,
    ) -> None:
        super().__init__()

        def down_conv(
            in_channels: int, out_channels: int, stride: Union[int, List[int]]
        ) -> List[nn.Module]:
            strides = stride if isinstance(stride, list) else [stride]
            layers = []
            for idx, s in enumerate(strides):
                kernel_size = s
                padding = 0
                if increment_kernel_size_on_down_conv:
                    if s % 2 != 0:
                        logging.warning(
                            f"increment_kernel_size_on_down_conv only works with even strides (not {s})"
                        )
                    else:
                        kernel_size += 1
                        padding = 1
                layers.extend(
                    [
                        layer_init(
                            nn.Conv2d(
                                in_channels if idx == 0 else out_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=s,
                                padding=padding,
                            ),
                            init_layers_orthogonal=init_layers_orthogonal,
                        ),
                        nn.GELU(),
                    ]
                )
            return layers

        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    *(
                        [
                            layer_init(
                                nn.Conv2d(
                                    in_channels, channels_per_level[0], 3, padding=1
                                ),
                                init_layers_orthogonal=init_layers_orthogonal,
                            ),
                            nn.GELU(),
                        ]
                        + [
                            ResidualBlock(
                                channels_per_level[0],
                                init_layers_orthogonal=init_layers_orthogonal,
                            )
                            for _ in range(encoder_residual_blocks_per_level[0])
                        ]
                    )
                )
            ]
        )
        for in_channels, channels, stride, num_residual_blocks in zip(
            channels_per_level[:-1],
            channels_per_level[1:],
            strides_per_level,
            encoder_residual_blocks_per_level[1:],
        ):
            self.encoders.append(
                nn.Sequential(
                    *(
                        down_conv(in_channels, channels, stride)
                        + [
                            ResidualBlock(
                                channels, init_layers_orthogonal=init_layers_orthogonal
                            )
                            for _ in range(num_residual_blocks)
                        ]
                    )
                )
            )

        def conv_transpose_2d(
            in_channels: int, out_channels: int, stride: Union[int, List[int]]
        ) -> List[nn.Module]:
            stride_factors = stride if isinstance(stride, list) else [stride]
            layers = []
            for idx, s in enumerate(stride_factors):
                layers.append(
                    layer_init(
                        nn.ConvTranspose2d(
                            in_channels if idx == 0 else out_channels,
                            out_channels,
                            kernel_size=s,
                            stride=s,
                        ),
                        init_layers_orthogonal=init_layers_orthogonal,
                    )
                )
                layers.append(nn.GELU())
            return layers

        conv_strides_per_level = conv_strides_per_level or strides_per_level
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    *conv_transpose_2d(
                        channels_per_level[-1],
                        channels_per_level[-2],
                        conv_strides_per_level[-1],
                    )
                )
            ]
        )
        for channels, out_channels, stride, num_residual_blocks in zip(
            reversed(channels_per_level[1:-1]),
            reversed(channels_per_level[:-2]),
            reversed(conv_strides_per_level[:-1]),
            reversed(decoder_residual_blocks_per_level[:-1]),
        ):
            self.decoders.append(
                nn.Sequential(
                    *(
                        [
                            ResidualBlock(
                                channels,
                                init_layers_orthogonal=init_layers_orthogonal,
                            )
                            for _ in range(num_residual_blocks)
                        ]
                        + conv_transpose_2d(channels, out_channels, stride)
                    )
                )
            )
        self.decoders.append(
            nn.Sequential(
                *[
                    ResidualBlock(
                        channels_per_level[0],
                        init_layers_orthogonal=init_layers_orthogonal,
                    )
                    for _ in range(decoder_residual_blocks_per_level[0])
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e_outs = [self.encoders[0](x)]
        for encoder in self.encoders[1:]:
            e_outs.append(encoder(e_outs[-1]))
        d_out = self.decoders[0](e_outs[-1])
        for e_out, decoder in zip(reversed(e_outs[:-1]), self.decoders[1:]):
            d_out = decoder(e_out + d_out)
        return d_out


class SqueezeUnetActorCriticNetwork(BackboneActorCritic):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_plane_space: Space,
        init_layers_orthogonal: bool = True,
        cnn_layers_init_orthogonal: Optional[bool] = None,
        channels_per_level: Optional[List[int]] = None,
        strides_per_level: Optional[List[Union[int, List[int]]]] = None,
        conv_strides_per_level: Optional[List[Union[int, List[int]]]] = None,
        encoder_residual_blocks_per_level: Optional[List[int]] = None,
        decoder_residual_blocks_per_level: Optional[List[int]] = None,
        num_additional_critics: int = 0,
        additional_critic_activation_functions: Optional[List[str]] = None,
        critic_channels: int = 64,
        increment_kernel_size_on_down_conv: bool = False,
    ) -> None:
        if cnn_layers_init_orthogonal is None:
            cnn_layers_init_orthogonal = False

        if channels_per_level is None:
            channels_per_level = [64, 128, 256]
        if strides_per_level is None:
            spl: List[Union[int, List[int]]] = [2] * (len(channels_per_level) - 1)
            strides_per_level = spl
        assert len(strides_per_level) == len(channels_per_level) - 1
        if encoder_residual_blocks_per_level is None:
            encoder_residual_blocks_per_level = [1] * len(channels_per_level)
        assert len(encoder_residual_blocks_per_level) == len(channels_per_level)
        if decoder_residual_blocks_per_level is None:
            decoder_residual_blocks_per_level = encoder_residual_blocks_per_level[:-1]
        assert (
            len(decoder_residual_blocks_per_level)
            == len(encoder_residual_blocks_per_level) - 1
        )
        assert isinstance(observation_space, Box)
        backbone = SqueezeUnetBackbone(
            observation_space.shape[0],  # type: ignore
            channels_per_level,
            strides_per_level,
            encoder_residual_blocks_per_level,
            decoder_residual_blocks_per_level,
            conv_strides_per_level=conv_strides_per_level,
            init_layers_orthogonal=cnn_layers_init_orthogonal,
            increment_kernel_size_on_down_conv=increment_kernel_size_on_down_conv,
        )
        super().__init__(
            observation_space,
            action_space,
            action_plane_space,
            backbone,
            channels_per_level[0],
            num_additional_critics=num_additional_critics,
            additional_critic_activation_functions=additional_critic_activation_functions,
            critic_channels=critic_channels,
            init_layers_orthogonal=init_layers_orthogonal,
            cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
            strides=strides_per_level,
        )
