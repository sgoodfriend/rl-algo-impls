from typing import Dict, List, NamedTuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box, Space

from rl_algo_impls.shared.module.normalization import normalization1d
from rl_algo_impls.shared.policy.actor_critic_network.backbone_actor_critic import (
    BackboneActorCritic,
)


class Grid2SeqTransformerNetwork(BackboneActorCritic):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_plane_space: Space,
        init_layers_orthogonal: bool = True,
        cnn_layers_init_orthogonal: Optional[bool] = None,
        encoder_embed_dim: int = 64,
        encoder_attention_heads: int = 4,
        encoder_feed_forward_dim: int = 256,
        encoder_layers: int = 4,
        num_additional_critics: int = 0,
        additional_critic_activation_functions: Optional[List[str]] = None,
        critic_channels: int = 64,
        critic_strides: Optional[List[Union[int, List[int]]]] = None,
        output_activation_fn: str = "identity",
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
        shared_critic_head: bool = False,
        normalization: str = "layer",
        add_position_features: bool = True,
        actor_head_kernel_size: int = 1,
        key_mask_empty_spaces: bool = True,
        identity_map_reordering: bool = True,
    ) -> None:
        if cnn_layers_init_orthogonal is None:
            cnn_layers_init_orthogonal = False

        if critic_strides is None:
            critic_strides = [4]

        assert isinstance(observation_space, Box)
        backbone = Grid2SeqTransformerBackbone(
            observation_space,
            encoder_embed_dim,
            encoder_attention_heads,
            encoder_feed_forward_dim,
            encoder_layers,
            normalization=normalization,
            add_position_features=add_position_features,
            key_mask_empty_spaces=key_mask_empty_spaces,
            identity_map_reordering=identity_map_reordering,
        )
        super().__init__(
            observation_space,
            action_space,
            action_plane_space,
            backbone,
            encoder_embed_dim,
            num_additional_critics=num_additional_critics,
            additional_critic_activation_functions=additional_critic_activation_functions,
            critic_channels=critic_channels,
            init_layers_orthogonal=init_layers_orthogonal,
            cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
            strides=critic_strides,
            output_activation_fn=output_activation_fn,
            subaction_mask=subaction_mask,
            shared_critic_head=shared_critic_head,
            normalization=normalization,
            actor_head_kernel_size=actor_head_kernel_size,
        )


class Grid2SeqTransformerBackbone(nn.Module):
    def __init__(
        self,
        observation_space: Box,
        encoder_embed_dim: int,
        encoder_attention_heads: int,
        encoder_feed_forward_dim: int,
        encoder_layers: int,
        normalization: str = "layer",
        add_position_features: bool = True,
        key_mask_empty_spaces: bool = True,
        identity_map_reordering: bool = True,
    ) -> None:
        super().__init__()
        channels, self.height, self.width = observation_space.shape

        self.add_position_features = add_position_features
        if add_position_features:
            y_pos = torch.linspace(-1, 1, self.height).tile(self.width, 1)  # [H, W]
            x_pos = torch.linspace(-1, 1, self.width).tile(self.height, 1).t()  # [H, W]
            position = torch.stack((y_pos, x_pos), dim=0)  # [2, H, W]

            self.register_buffer("position", position)
            channels += 2
        self.key_mask_empty_spaces = key_mask_empty_spaces

        self.encoder_embed_dim = encoder_embed_dim
        self.embedding_layer = nn.Sequential(
            *[
                nn.Conv2d(channels, encoder_embed_dim, 1),
                nn.ReLU(),
            ]
        )
        self.encoding_layers = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    encoder_embed_dim,
                    encoder_attention_heads,
                    encoder_feed_forward_dim,
                    normalization,
                    identity_map_reordering,
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.key_mask_empty_spaces:
            NO_UNIT_TYPE_IDX = 6
            IN_BOUNDS_IDX = 58
            mask_cols = (
                x[:, (NO_UNIT_TYPE_IDX, IN_BOUNDS_IDX)].flatten(2).bool()
            )  # bool[B, 2, H*W]
            key_padding_mask = mask_cols.all(dim=1)  # bool[B, H*W]
        else:
            key_padding_mask = None

        if self.add_position_features:
            x = torch.cat((x, self.position.expand(x.size(0), -1, -1, -1)), dim=1)
        x = self.embedding_layer(x)  # [B, C, H, W] -> [B, embed_dim, H, W]
        x = x.flatten(2).permute(0, 2, 1)  # [B, embed_dim, H, W] -> [B, H*W, embed_dim]
        x, _ = self.encoding_layers(TransformerEncoderForwardArgs(x, key_padding_mask))
        x = x.permute(0, 2, 1).reshape(
            -1, self.encoder_embed_dim, self.height, self.width
        )  # [B, H*W, embed_dim] -> [B, embed_dim, H, W]
        return x


class TransformerEncoderForwardArgs(NamedTuple):
    x: torch.Tensor
    key_padding_mask: Optional[torch.Tensor]


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        attention_heads: int,
        feed_forward_dim: int,
        normalization: str = "layer",
        identity_map_reordering: bool = True,
    ):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim, attention_heads, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, embed_dim),
        )

        self.attention_normalization = normalization1d(normalization, embed_dim)
        self.feed_forward_normalization = normalization1d(normalization, embed_dim)
        self.identity_map_reordering = identity_map_reordering

    def forward(
        self, args: TransformerEncoderForwardArgs
    ) -> TransformerEncoderForwardArgs:
        x, key_padding_mask = args
        if self.identity_map_reordering:
            attention_input = self.attention_normalization(x)
            attention_output, _ = self.multihead_attention(
                attention_input,
                attention_input,
                attention_input,
                key_padding_mask=key_padding_mask,
            )
            x = x + attention_output

            ff_input = self.feed_forward_normalization(x)
            ff_output = self.feed_forward(ff_input)
            x = x + ff_output
        else:
            attention_output, _ = self.multihead_attention(
                x, x, x, key_padding_mask=key_padding_mask
            )
            x = x + attention_output
            x = self.attention_normalization(x)
            ff_output = self.feed_forward(x)
            x = x + ff_output
            x = self.feed_forward_normalization(x)
        return TransformerEncoderForwardArgs(x, key_padding_mask)
