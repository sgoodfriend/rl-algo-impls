from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box
from gym.spaces import Dict as DictSpace
from gym.spaces import MultiDiscrete, Space

from rl_algo_impls.shared.actor import pi_forward
from rl_algo_impls.shared.actor.gridnet import GridnetDistribution
from rl_algo_impls.shared.actor.gridnet_decoder import Transpose
from rl_algo_impls.shared.module.stack import HStack
from rl_algo_impls.shared.module.utils import layer_init
from rl_algo_impls.shared.policy.actor_critic_network.network import (
    ACNForward,
    ActorCriticNetwork,
)
from rl_algo_impls.shared.policy.policy import ACTIVATION


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
    ) -> None:
        super().__init__()
        self.pool_conv = layer_init(
            nn.Conv2d(channels, pooled_channels, 4, stride=4),
            init_layers_orthogonal=init_layers_orthogonal,
        )
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
        return x + self.up_conv(self.res_blocks(self.pool_conv(x)))


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
        )
        self.out_block = nn.Sequential(
            *[
                ResidualBlock(channels, init_layers_orthogonal=init_layers_orthogonal)
                for _ in range(out_num_res_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_block(self.double_cone(self.in_block(x)))


class DoubleConeActorCritic(ActorCriticNetwork):
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
    ) -> None:
        if cnn_layers_init_orthogonal is None:
            cnn_layers_init_orthogonal = False
        if num_additional_critics and not additional_critic_activation_functions:
            additional_critic_activation_functions = [
                "identity"
            ] * num_additional_critics
        super().__init__()
        assert isinstance(observation_space, Box)
        assert isinstance(action_plane_space, MultiDiscrete)
        self.range_size = np.max(observation_space.high) - np.min(observation_space.low)  # type: ignore
        self.action_vec = action_plane_space.nvec  # type: ignore
        if isinstance(action_space, DictSpace):
            action_space_per_position = action_space["per_position"]  # type: ignore
            self.pick_vec = action_space["pick_position"].nvec  # type: ignore
        elif isinstance(action_space, MultiDiscrete):
            action_space_per_position = action_space
            self.pick_vec = None
        else:
            raise ValueError(
                f"action_space {action_space.__class__.__name__} must be MultiDiscrete or gym Dict of MultiDiscrete"
            )

        self.map_size = len(action_space_per_position.nvec) // len(action_plane_space.nvec)  # type: ignore

        self.backbone = DoubleConeBackbone(
            observation_space.shape[0],  # type: ignore
            backbone_channels,
            pooled_channels,
            init_layers_orthogonal=cnn_layers_init_orthogonal,
            in_num_res_blocks=in_num_res_blocks,
            cone_num_res_blocks=cone_num_res_blocks,
            out_num_res_blocks=out_num_res_blocks,
        )
        self.actor_head = nn.Sequential(
            *[
                layer_init(
                    nn.Conv2d(
                        in_channels=backbone_channels,
                        out_channels=self.action_vec.sum()
                        + (len(self.pick_vec) if self.pick_vec else 0),
                        kernel_size=3,
                        padding=1,
                    ),
                    init_layers_orthogonal=init_layers_orthogonal,
                    std=0.01,
                ),
                Transpose((0, 2, 3, 1)),
            ]
        )

        def critic_head(output_activation_name: str = "identity") -> nn.Module:
            return nn.Sequential(
                *[
                    layer_init(
                        nn.Conv2d(
                            backbone_channels, critic_channels, 3, stride=2, padding=1
                        ),
                        init_layers_orthogonal=cnn_layers_init_orthogonal,
                    ),
                    nn.GELU(),
                    layer_init(
                        nn.Conv2d(
                            critic_channels, critic_channels, 3, stride=2, padding=1
                        ),
                        init_layers_orthogonal=cnn_layers_init_orthogonal,
                    ),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    layer_init(
                        nn.Linear(critic_channels, critic_channels),
                        init_layers_orthogonal=init_layers_orthogonal,
                    ),
                    nn.GELU(),
                    layer_init(
                        nn.Linear(critic_channels, 1),
                        init_layers_orthogonal=init_layers_orthogonal,
                        std=1.0,
                    ),
                    ACTIVATION[output_activation_name](),
                ]
            )

        self.critic_heads = HStack(
            [
                critic_head(act_fn_name)
                for act_fn_name in ["identity"]
                + (additional_critic_activation_functions or [])
            ]
        )

    def _preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        return obs.float() / self.range_size

    def _distribution_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
    ) -> ACNForward:
        assert (
            action_masks is not None
        ), f"No mask case unhandled in {self.__class__.__name__}"

        import time

        s_ts = time.perf_counter()
        o = self._preprocess(obs)
        x = self.backbone(o)
        logits = self.actor_head(x)
        pi = GridnetDistribution(
            int(np.prod(o.shape[-2:])), self.action_vec, logits, action_masks
        )
        e_ts = time.perf_counter()
        d_ms = (e_ts - s_ts) * 1000
        if d_ms >= 100:
            import logging

            logging.warn(f"Network took too long: {int(d_ms)}ms")

        v = self.critic_heads(x)
        if v.shape[-1] == 1:
            v.squeeze(-1)

        return ACNForward(pi_forward(pi, action), v)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        o = self._preprocess(obs)
        x = self.backbone(o)
        v = self.critic_heads(x)
        if v.shape[-1] == 1:
            v.squeeze(-1)
        return v

    def reset_noise(self, batch_size: Optional[int] = None) -> None:
        pass

    @property
    def action_shape(self) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
        per_position_action_shape = (self.map_size, len(self.action_vec))
        if self.pick_vec:
            return {
                "per_position": per_position_action_shape,
                "pick_position": (len(self.pick_vec),),
            }
        return per_position_action_shape

    @property
    def value_shape(self) -> Tuple[int, ...]:
        if len(self.critic_heads) > 1:
            return (len(self.critic_heads),)
        else:
            return ()
