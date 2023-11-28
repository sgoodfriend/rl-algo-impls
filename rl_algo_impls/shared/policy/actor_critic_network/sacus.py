from typing import Dict, List, Optional, Union

import torch
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import MultiDiscrete, Space

from rl_algo_impls.shared.policy.actor_critic_network.shared_ac_backboned_network import (
    SplitActorCriticBackbone,
    SplitActorCriticBackbonedNetwork,
    SplitActorCriticBackboneOutput,
)
from rl_algo_impls.shared.policy.actor_critic_network.squeeze_unet import (
    SqueezeUnetBackbone,
)


class SplitActorCriticUShapedNetwork(SplitActorCriticBackbonedNetwork):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_plane_space: Space,
        init_layers_orthogonal: bool = True,
        cnn_layers_init_orthogonal: Optional[bool] = None,
        channels_per_level: Optional[List[int]] = None,
        strides_per_level: Optional[List[Union[int, List[int]]]] = None,
        deconv_strides_per_level: Optional[List[Union[int, List[int]]]] = None,
        encoder_residual_blocks_per_level: Optional[List[int]] = None,
        decoder_residual_blocks_per_level: Optional[List[int]] = None,
        num_additional_critics: int = 0,
        additional_critic_activation_functions: Optional[List[str]] = None,
        critic_channels: int = 128,
        increment_kernel_size_on_down_conv: bool = False,
        output_activation_fn: str = "identity",
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
        shared_critic_head: bool = False,
    ):
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

        backbone = SplitActorCriticUShapedBackbone(
            observation_space.shape[0],
            channels_per_level,
            strides_per_level,
            encoder_residual_blocks_per_level,
            decoder_residual_blocks_per_level,
            deconv_strides_per_level=deconv_strides_per_level,
            init_layers_orthogonal=cnn_layers_init_orthogonal,
            increment_kernel_size_on_down_conv=increment_kernel_size_on_down_conv,
        )

        assert isinstance(action_plane_space, MultiDiscrete)
        assert isinstance(action_space, (DictSpace, MultiDiscrete))
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
            output_activation_fn=output_activation_fn,
            subaction_mask=subaction_mask,
            shared_critic_head=shared_critic_head,
        )


class SplitActorCriticUShapedBackbone(SplitActorCriticBackbone, SqueezeUnetBackbone):
    def forward(self, x: torch.Tensor) -> SplitActorCriticBackboneOutput:
        e_outs = []
        for idx, encoder in enumerate(self.encoders):
            if idx == 0:
                e_outs.append(encoder(x))
            else:
                e_outs.append(encoder(e_outs[-1]))
        d_out = torch.zeros_like(e_outs[-1])
        for e_out, decoder in zip(reversed(e_outs), self.decoders):
            d_out = decoder(e_out + d_out)
        return SplitActorCriticBackboneOutput(
            actor_attachment=d_out,
            critic_attachment=e_outs[-1],
        )

    def value_head_input(self, x: torch.Tensor) -> torch.Tensor:
        e_outs = []
        for idx, encoder in enumerate(self.encoders):
            if idx == 0:
                e_outs.append(encoder(x))
            else:
                e_outs.append(encoder(e_outs[-1]))
        return e_outs[-1]
