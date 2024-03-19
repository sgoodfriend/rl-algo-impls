from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, MultiDiscrete, Space

from rl_algo_impls.shared.actor import pi_forward
from rl_algo_impls.shared.actor.gridnet import ValueDependentMask
from rl_algo_impls.shared.module.channelwise_activation import ChannelwiseActivation
from rl_algo_impls.shared.module.normalization import (
    NormalizationMethod,
    normalization1d,
)
from rl_algo_impls.shared.module.utils import mlp
from rl_algo_impls.shared.policy.actor_critic_network.grid2seq_transformer import (
    TransformerEncoderForwardArgs,
    TransformerEncoderLayer,
    empty_spaces_mask,
)
from rl_algo_impls.shared.policy.actor_critic_network.network import (
    ACNForward,
    ActorCriticNetwork,
)
from rl_algo_impls.shared.policy.grid2entity_distribution import Grid2EntityDistribution
from rl_algo_impls.shared.policy.policy import ACTIVATION


class BackboneForwardReturn(NamedTuple):
    x: torch.Tensor  # Float[B, S, E]
    key_padding_mask: torch.Tensor  # Bool[B, S]
    keep_mask: torch.Tensor  # Bool[B, H*W]
    n_keep: torch.Tensor  # Int[B]


class Grid2EntityTransformerNetwork(ActorCriticNetwork):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_plane_space: Space,
        init_layers_orthogonal: bool = True,
        hidden_embedding_dims: Optional[List[int]] = None,
        encoder_embed_dim: int = 64,
        encoder_attention_heads: int = 4,
        encoder_feed_forward_dim: int = 256,
        encoder_layers: int = 4,
        num_additional_critics: int = 0,
        additional_critic_activation_functions: Optional[List[str]] = None,
        hidden_critic_dims: Optional[List[int]] = None,
        hidden_actor_dims: Optional[List[int]] = None,
        output_activation_fn: str = "identity",
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
        normalization: str = "layer",
        post_backbone_normalization: Optional[str] = "layer",
        add_position_features: bool = True,
    ) -> None:
        if hidden_embedding_dims is None:
            hidden_embedding_dims = []
        if hidden_critic_dims is None:
            hidden_critic_dims = []
        if hidden_actor_dims is None:
            hidden_actor_dims = []
        if num_additional_critics and not additional_critic_activation_functions:
            additional_critic_activation_functions = [
                "identity"
            ] * num_additional_critics

        assert (
            normalization.upper() in NormalizationMethod.__members__
        ), f"Invalid normalization method {normalization}"

        super().__init__()

        assert isinstance(observation_space, Box)
        assert isinstance(action_plane_space, MultiDiscrete)
        self.range_size = np.max(observation_space.high) - np.min(observation_space.low)
        self.action_vec = action_plane_space.nvec

        assert isinstance(
            action_space, MultiDiscrete
        ), f"Only MultiDiscrete action space supported, not {action_space.__class__.__name__}"

        channels, self.height, self.width = observation_space.shape

        self.add_position_features = add_position_features
        if add_position_features:
            y_pos = torch.linspace(-1, 1, self.height).tile(self.width, 1)  # [H, W]
            x_pos = torch.linspace(-1, 1, self.width).tile(self.height, 1).t()  # [H, W]
            position = torch.stack((y_pos, x_pos), dim=0)  # [2, H, W]

            self.register_buffer("position", position)
            channels += 2

        embedding_layer_sizes = [channels, *hidden_embedding_dims, encoder_embed_dim]
        self.embedding_layer = mlp(
            embedding_layer_sizes,
            nn.GELU,
            output_activation=nn.GELU(),
            init_layers_orthogonal=init_layers_orthogonal,
        )

        self.backbone = TransformerEncoderBackbone(
            encoder_embed_dim,
            encoder_attention_heads,
            encoder_feed_forward_dim,
            encoder_layers,
            normalization=normalization,
        )

        self.post_backbone_normalization = (
            normalization1d(post_backbone_normalization, encoder_embed_dim)
            if post_backbone_normalization
            else None
        )

        actor_layer_sizes = [
            encoder_embed_dim,
            *hidden_actor_dims,
            self.action_vec.sum(),
        ]
        self.actor_head = mlp(
            actor_layer_sizes,
            nn.GELU,
            init_layers_orthogonal=init_layers_orthogonal,
            final_layer_gain=0.01,
        )
        self.subaction_mask = (
            ValueDependentMask.from_reference_index_to_index_to_value(subaction_mask)
            if subaction_mask
            else None
        )

        output_activations = [
            ACTIVATION[act_fn_name]()
            for act_fn_name in [output_activation_fn]
            + (additional_critic_activation_functions or [])
        ]
        critic_layer_sizes = [
            encoder_embed_dim,
            *hidden_critic_dims,
            len(output_activations),
        ]
        self.critic_head = mlp(
            critic_layer_sizes,
            nn.GELU,
            output_activation=ChannelwiseActivation(output_activations),
            init_layers_orthogonal=init_layers_orthogonal,
            final_layer_gain=1.0,
        )

    def _preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        return obs.float() / self.range_size

    def _backbone_forward(self, obs: torch.Tensor) -> BackboneForwardReturn:
        x = self._preprocess(obs)  # Float[B, C, H, W]
        if self.add_position_features:
            x = torch.cat((x, self.position.expand(x.size(0), -1, -1, -1)), dim=1)

        x = x.flatten(2).permute(0, 2, 1)  # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]

        keep_mask = ~empty_spaces_mask(x)  # Bool[B, H*W]
        n_keep = keep_mask.sum(dim=1)  # Int[B]
        s_dim = n_keep.max().item()
        assert isinstance(s_dim, int)

        entities = x[keep_mask]  # Float[Sum(n_keep), C]
        row_indices = torch.arange(x.size(0), device=x.device).repeat_interleave(
            n_keep
        )  # Int[Sum(n_keep)]
        cumulative_true_counts = (
            keep_mask.cumsum(dim=1).masked_select(keep_mask) - 1
        )  # Int[Sum(n_keep)]
        x = torch.zeros(
            x.size(0), s_dim, x.size(2), dtype=x.dtype, device=x.device
        )  # Float[B, S, C]
        x[row_indices, cumulative_true_counts] = entities

        key_padding_mask = torch.full(
            (x.size(0), s_dim),
            fill_value=True,
            dtype=keep_mask.dtype,
            device=keep_mask.device,
        )  # Bool[B, S]
        key_padding_mask[row_indices, cumulative_true_counts] = False

        x = self.embedding_layer(x)  # [B, S, C] -> [B, S, E]
        x = self.backbone(x, key_padding_mask=key_padding_mask)
        if self.post_backbone_normalization is not None:
            x = self.post_backbone_normalization(x)
        return BackboneForwardReturn(x, key_padding_mask, keep_mask, n_keep)

    def _distribution_and_value(
        self,
        obs: torch.Tensor,  # Float[B, C, H, W]
        action: Optional[torch.Tensor] = None,  # Optional[Int[B, H*W, A]]
        action_masks: Optional[torch.Tensor] = None,  # Bool[B, H*W, A]
    ) -> ACNForward:
        assert (
            action_masks is not None
        ), f"No mask case unhandled in {self.__class__.__name__}"
        x, key_padding_mask, keep_mask, n_keep = self._backbone_forward(
            obs
        )  # [B, S, E], [B, S], [B, H*W], [B]

        _mask_squash = torch.zeros(
            action_masks.size(0),
            x.size(1),
            action_masks.size(2),
            dtype=action_masks.dtype,
            device=action_masks.device,
        )  # Bool[B, S, A]
        for i in range(action_masks.size(0)):
            _mask_squash[i, : n_keep[i], :] = action_masks[i, keep_mask[i], :]
        action_masks = _mask_squash

        action_logits = self.actor_head(x)  # -> [B, S, A]
        pi = Grid2EntityDistribution(
            keep_mask,
            n_keep,
            self.action_vec,
            action_logits,
            action_masks,
            subaction_mask=self.subaction_mask,
        )

        v_input = torch.where(key_padding_mask.unsqueeze(-1), 0, x).sum(dim=1) / (
            n_keep.unsqueeze(-1) + 1e-6
        )  # [B, S, E] -> [B, E]
        v = self.critic_head(v_input)  # -> [B, V]
        if v.shape[-1] == 1:
            v = v.squeeze(-1)

        return ACNForward(pi_forward(pi, action), v)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        x, key_padding_mask, _, n_keep = self._backbone_forward(obs)
        v_input = torch.where(key_padding_mask.unsqueeze(-1), 0, x).sum(dim=1) / (
            n_keep.unsqueeze(-1) + 1e-6
        )  # [B, S, E] -> [B, E]
        v = self.critic_head(v_input)  # -> [B, V]
        if v.shape[-1] == 1:
            v = v.squeeze(-1)
        return v

    def reset_noise(self, batch_size: int) -> None:
        pass

    def freeze(
        self,
        freeze_policy_head: bool,
        freeze_value_head: bool,
        freeze_backbone: bool = True,
    ) -> None:
        for p in self.embedding_layer.parameters():
            p.requires_grad = not freeze_backbone
        for p in self.backbone.parameters():
            p.requires_grad = not freeze_backbone

        for p in self.actor_head.parameters():
            p.requires_grad = not freeze_policy_head

        for p in self.critic_head.parameters():
            p.requires_grad = not freeze_value_head


class TransformerEncoderBackbone(nn.Module):
    def __init__(
        self,
        encoder_embed_dim: int,
        encoder_attention_heads: int,
        encoder_feed_forward_dim: int,
        encoder_layers: int,
        normalization: str = "layer",
    ) -> None:
        super().__init__()

        self.encoders = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    encoder_embed_dim,
                    encoder_attention_heads,
                    encoder_feed_forward_dim,
                    normalization=normalization,
                    identity_map_reordering=True,
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x, _ = self.encoders(
            TransformerEncoderForwardArgs(x, key_padding_mask=key_padding_mask)
        )
        return x
