from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import MultiDiscrete

from rl_algo_impls.shared.actor import pi_forward
from rl_algo_impls.shared.actor.gridnet import GridnetDistribution, ValueDependentMask
from rl_algo_impls.shared.actor.gridnet_decoder import Transpose
from rl_algo_impls.shared.module.adaptive_avg_max_pool import AdaptiveAvgMaxPool2d
from rl_algo_impls.shared.module.channelwise_activation import ChannelwiseActivation
from rl_algo_impls.shared.module.stack import HStack
from rl_algo_impls.shared.module.utils import layer_init
from rl_algo_impls.shared.policy.actor_critic_network.network import (
    ACNForward,
    ActorCriticNetwork,
)
from rl_algo_impls.shared.policy.policy import ACTIVATION


class SplitActorCriticBackboneOutput(NamedTuple):
    actor_attachment: torch.Tensor
    critic_attachment: torch.Tensor


class SplitActorCriticBackbone(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> SplitActorCriticBackboneOutput:
        ...

    @abstractmethod
    def value_head_input(self, x: torch.Tensor) -> torch.Tensor:
        ...


class SplitActorCriticBackbonedNetwork(ActorCriticNetwork):
    def __init__(
        self,
        observation_space: Box,
        action_space: Union[DictSpace, MultiDiscrete],
        action_plane_space: MultiDiscrete,
        backbone: SplitActorCriticBackbone,
        backbone_out_channels: int,
        num_additional_critics: int = 0,
        additional_critic_activation_functions: Optional[List[str]] = None,
        critic_channels: int = 64,
        init_layers_orthogonal: bool = True,
        output_activation_fn: str = "identity",
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
        shared_critic_head: bool = False,
        critic_avg_max_pool: bool = False,
    ) -> None:
        if num_additional_critics and not additional_critic_activation_functions:
            additional_critic_activation_functions = [
                "identity"
            ] * num_additional_critics

        super().__init__()
        self.backbone = backbone

        self.range_size = np.max(observation_space.high) - np.min(observation_space.low)
        self.action_vec = action_plane_space.nvec

        if isinstance(action_space, DictSpace):
            action_space_per_position = action_space["per_position"]
            pick_position_space = action_space["pick_position"]
            assert isinstance(pick_position_space, MultiDiscrete)
            self.pick_vec = pick_position_space.nvec
        elif isinstance(action_space, MultiDiscrete):
            action_space_per_position = action_space
            self.pick_vec = None
        else:
            raise ValueError(
                f"action_space {action_space.__class__.__name__} must be MultiDiscrete or gymnasium Dict of MultiDiscrete"
            )

        self.subaction_mask = subaction_mask
        self.shared_critic_head = shared_critic_head

        assert isinstance(action_space_per_position, MultiDiscrete)
        self.map_size = len(action_space_per_position.nvec) // len(
            action_plane_space.nvec
        )

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

        def critic_head(
            output_activation_layer: nn.Module, num_output_channels: int = 1
        ) -> nn.Module:
            linear_in_channels = critic_channels
            if critic_avg_max_pool:
                linear_in_channels *= 2
                pool_layer = AdaptiveAvgMaxPool2d(pool_output_size=1)
            else:
                pool_layer = nn.AdaptiveAvgPool2d(1)
            return nn.Sequential(
                pool_layer,
                nn.Flatten(),
                layer_init(
                    nn.Linear(linear_in_channels, critic_channels),
                    init_layers_orthogonal=init_layers_orthogonal,
                ),
                nn.GELU(),
                layer_init(
                    nn.Linear(critic_channels, num_output_channels),
                    init_layers_orthogonal=init_layers_orthogonal,
                    std=1.0,
                ),
                output_activation_layer,
            )

        output_activations = [
            ACTIVATION[act_fn_name]()
            for act_fn_name in [output_activation_fn]
            + (additional_critic_activation_functions or [])
        ]
        self._critic_features = len(output_activations)
        if self.shared_critic_head:
            self.critic_heads = critic_head(
                ChannelwiseActivation(output_activations),
                num_output_channels=len(output_activations),
            )
        else:
            self.critic_heads = HStack(
                [critic_head(activation) for activation in output_activations]
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
        actor_x, critic_x = self.backbone(o)
        logits = self.actor_head(actor_x)
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

        v = self.critic_heads(critic_x)
        if v.shape[-1] == 1:
            v = v.squeeze(-1)

        return ACNForward(pi_forward(pi, action), v)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        o = self._preprocess(obs)
        critic_x = self.backbone.value_head_input(o)
        v = self.critic_heads(critic_x)
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
        if self._critic_features > 1:
            return (self._critic_features,)
        return ()

    def freeze(
        self,
        freeze_policy_head: bool,
        freeze_value_head: bool,
        freeze_backbone: bool = True,
    ) -> None:
        for param in self.actor_head.parameters():
            param.requires_grad = not freeze_policy_head
        for param in self.critic_heads.parameters():
            param.requires_grad = not freeze_value_head
        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone
