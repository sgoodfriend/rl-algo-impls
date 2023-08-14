from typing import Dict, List, Optional

import torch
import torch.nn as nn
from gym.spaces import Box, Space

from rl_algo_impls.shared.module.utils import layer_init
from rl_algo_impls.shared.policy.actor_critic_network.backbone_actor_critic import (
    BackboneActorCritic,
)


class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        init_layers_orthogonal: bool = False,
    ) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        bottleneck_channels = in_channels // reduction_ratio
        self.fc = nn.Sequential(
            *[
                layer_init(
                    nn.Linear(in_channels, bottleneck_channels, bias=False),
                    init_layers_orthogonal=init_layers_orthogonal,
                ),
                nn.GELU(),
                layer_init(
                    nn.Linear(bottleneck_channels, in_channels, bias=False),
                    init_layers_orthogonal=init_layers_orthogonal,
                ),
                nn.Sigmoid(),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()
        pooled = self.avg_pool(x).view(batch_size, channels)
        scale = self.fc(pooled).view(batch_size, channels, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        init_layers_orthogonal: bool = False,
    ) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            layer_init(
                nn.Conv2d(channels, channels, 3, padding=1),
                init_layers_orthogonal=init_layers_orthogonal,
            ),
            nn.GELU(),
            layer_init(
                nn.Conv2d(channels, channels, 3, padding=1),
                init_layers_orthogonal=init_layers_orthogonal,
            ),
            SqueezeExcitation(channels, init_layers_orthogonal=init_layers_orthogonal),
        )
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(x + self.residual(x))


class DoubleConeBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        pooled_channels: int,
        num_residual_blocks: int = 6,
        init_layers_orthogonal: bool = False,
        gelu_pool_conv: bool = True,
    ) -> None:
        super().__init__()
        self.pool_conv = layer_init(
            nn.Conv2d(channels, pooled_channels, 4, stride=4),
            init_layers_orthogonal=init_layers_orthogonal,
        )
        self.gelu = nn.GELU() if gelu_pool_conv else nn.Identity()
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    pooled_channels, init_layers_orthogonal=init_layers_orthogonal
                )
                for _ in range(num_residual_blocks)
            ]
        )
        intermediate_channels = (channels + pooled_channels) // 2
        self.up_conv = nn.Sequential(
            *[
                layer_init(
                    nn.ConvTranspose2d(
                        pooled_channels, intermediate_channels, 2, stride=2
                    ),
                    init_layers_orthogonal=init_layers_orthogonal,
                ),
                nn.GELU(),
                layer_init(
                    nn.ConvTranspose2d(intermediate_channels, channels, 2, stride=2),
                    init_layers_orthogonal=init_layers_orthogonal,
                ),
                nn.GELU(),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up_conv(self.res_blocks(self.gelu(self.pool_conv(x))))


class DoubleConeBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        pooled_channels: int,
        init_layers_orthogonal: bool = False,
        in_num_res_blocks: int = 4,
        cone_num_res_blocks: int = 6,
        out_num_res_blocks: int = 4,
        gelu_pool_conv: bool = True,
    ) -> None:
        super().__init__()
        self.in_block = nn.Sequential(
            *(
                [
                    layer_init(
                        nn.Conv2d(in_channels, channels, 3, padding=1),
                        init_layers_orthogonal=init_layers_orthogonal,
                    ),
                    nn.GELU(),
                ]
                + [
                    ResidualBlock(
                        channels, init_layers_orthogonal=init_layers_orthogonal
                    )
                    for _ in range(in_num_res_blocks)
                ]
            )
        )
        self.double_cone = DoubleConeBlock(
            channels,
            pooled_channels,
            num_residual_blocks=cone_num_res_blocks,
            init_layers_orthogonal=init_layers_orthogonal,
            gelu_pool_conv=gelu_pool_conv,
        )
        self.out_block = nn.Sequential(
            *[
                ResidualBlock(channels, init_layers_orthogonal=init_layers_orthogonal)
                for _ in range(out_num_res_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_block(self.double_cone(self.in_block(x)))


class DoubleConeActorCritic(BackboneActorCritic):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_plane_space: Space,
        init_layers_orthogonal: bool = True,
        cnn_layers_init_orthogonal: Optional[bool] = None,
        backbone_channels: int = 128,
        pooled_channels: int = 512,
        critic_channels: int = 64,
        in_num_res_blocks: int = 4,
        cone_num_res_blocks: int = 6,
        out_num_res_blocks: int = 4,
        num_additional_critics: int = 0,
        additional_critic_activation_functions: Optional[List[str]] = None,
        gelu_pool_conv: bool = True,
        output_activation_fn: str = "identity",
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
    ) -> None:
        if cnn_layers_init_orthogonal is None:
            cnn_layers_init_orthogonal = False
        assert isinstance(observation_space, Box)
        backcone = DoubleConeBackbone(
            observation_space.shape[0],  # type: ignore
            backbone_channels,
            pooled_channels,
            init_layers_orthogonal=cnn_layers_init_orthogonal,
            in_num_res_blocks=in_num_res_blocks,
            cone_num_res_blocks=cone_num_res_blocks,
            out_num_res_blocks=out_num_res_blocks,
            gelu_pool_conv=gelu_pool_conv,
        )
        super().__init__(
            observation_space,
            action_space,
            action_plane_space,
            backcone,
            backbone_channels,
            num_additional_critics=num_additional_critics,
            additional_critic_activation_functions=additional_critic_activation_functions,
            critic_channels=critic_channels,
            init_layers_orthogonal=init_layers_orthogonal,
            cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
            output_activation_fn=output_activation_fn,
            subaction_mask=subaction_mask,
        )
