from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box
from gym.spaces import Dict as DictSpace
from gym.spaces import MultiDiscrete, Space

from rl_algo_impls.shared.actor import pi_forward
from rl_algo_impls.shared.actor.gridnet import GridnetDistribution, ValueDependentMask
from rl_algo_impls.shared.actor.gridnet_decoder import Transpose
from rl_algo_impls.shared.module.stack import HStack
from rl_algo_impls.shared.module.utils import layer_init
from rl_algo_impls.shared.policy.actor_critic_network.network import (
    ACNForward,
    ActorCriticNetwork,
)
from rl_algo_impls.shared.policy.policy import ACTIVATION


class BackboneActorCritic(ActorCriticNetwork):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_plane_space: Space,
        backbone: nn.Module,
        backbone_out_channels: int,
        num_additional_critics: int = 0,
        additional_critic_activation_functions: Optional[List[str]] = None,
        critic_channels: int = 64,
        init_layers_orthogonal: bool = True,
        cnn_layers_init_orthogonal: bool = False,
        strides: Sequence[Union[int, Sequence[int]]] = (2, 2),
        output_activation_fn: str = "identity",
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
    ):
        if num_additional_critics and not additional_critic_activation_functions:
            additional_critic_activation_functions = [
                "identity"
            ] * num_additional_critics

        super().__init__()
        self.backbone = backbone

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
        self.subaction_mask = subaction_mask

        self.map_size = len(action_space_per_position.nvec) // len(action_plane_space.nvec)  # type: ignore

        self.actor_head = nn.Sequential(
            *[
                layer_init(
                    nn.Conv2d(
                        in_channels=backbone_out_channels,
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
            def down_conv(
                in_channels: int, out_channels: int, stride: int
            ) -> List[nn.Module]:
                kernel_size = max(3, stride)
                return [
                    layer_init(
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                            padding=1 if kernel_size % 2 else 0,
                        ),
                        init_layers_orthogonal=cnn_layers_init_orthogonal,
                    ),
                    nn.GELU(),
                ]

            flattened_strides = []
            for s in strides:
                if isinstance(s, list):
                    flattened_strides.extend(s)
                else:
                    flattened_strides.append(s)

            down_convs = down_conv(
                backbone_out_channels, critic_channels, flattened_strides[0]
            )
            for s in flattened_strides[1:]:
                down_convs.extend(down_conv(critic_channels, critic_channels, s))
            return nn.Sequential(
                *(
                    down_convs
                    + [
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
            )

        self.critic_heads = HStack(
            [
                critic_head(act_fn_name)
                for act_fn_name in [output_activation_fn]
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

        o = self._preprocess(obs)
        x = self.backbone(o)
        logits = self.actor_head(x)
        pi = GridnetDistribution(
            int(np.prod(o.shape[-2:])),
            self.action_vec,
            logits,
            action_masks,
            subaction_mask=ValueDependentMask.from_reference_index_to_index_to_value(
                self.subaction_mask
            )
            if self.subaction_mask
            else None,
        )

        v = self.critic_heads(x)
        if v.shape[-1] == 1:
            v = v.squeeze(-1)

        return ACNForward(pi_forward(pi, action), v)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        o = self._preprocess(obs)
        x = self.backbone(o)
        v = self.critic_heads(x)
        if v.shape[-1] == 1:
            v = v.squeeze(-1)
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

    def freeze(
        self,
        freeze_policy_head: bool,
        freeze_value_head: bool,
        freeze_backbone: bool = True,
    ) -> None:
        for p in self.actor_head.parameters():
            p.requires_grad = not freeze_policy_head
        for p in self.critic_heads.parameters():
            p.requires_grad = not freeze_value_head
        for p in self.backbone.parameters():
            p.requires_grad = not freeze_backbone
