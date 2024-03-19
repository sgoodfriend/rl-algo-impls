from abc import abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import gymnasium
import numpy as np
import torch
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace

from rl_algo_impls.shared.policy.abstract_policy import Step
from rl_algo_impls.shared.policy.actor_critic_network import (
    ConnectedTrioActorCriticNetwork,
    SeparateActorCriticNetwork,
    UNetActorCriticNetwork,
)
from rl_algo_impls.shared.policy.actor_critic_network.backbone_actor_critic import (
    BackboneActorCritic,
)
from rl_algo_impls.shared.policy.actor_critic_network.double_cone import (
    DoubleConeActorCritic,
)
from rl_algo_impls.shared.policy.actor_critic_network.grid2entity_transformer import (
    Grid2EntityTransformerNetwork,
)
from rl_algo_impls.shared.policy.actor_critic_network.grid2seq_transformer import (
    Grid2SeqTransformerNetwork,
)
from rl_algo_impls.shared.policy.actor_critic_network.sacus import (
    SplitActorCriticUShapedNetwork,
)
from rl_algo_impls.shared.policy.actor_critic_network.squeeze_unet import (
    SqueezeUnetActorCriticNetwork,
)
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.tensor_utils import NumpyOrDict, tensor_to_numpy
from rl_algo_impls.shared.vec_env.action_shape import ActionShape
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces
from rl_algo_impls.wrappers.vector_wrapper import ObsType


class ACForward(NamedTuple):
    logp_a: torch.Tensor
    entropy: torch.Tensor
    v: torch.Tensor


FEAT_EXT_FILE_NAME = "feat_ext.pt"
V_FEAT_EXT_FILE_NAME = "v_feat_ext.pt"
PI_FILE_NAME = "pi.pt"
V_FILE_NAME = "v.pt"
ActorCriticSelf = TypeVar("ActorCriticSelf", bound="ActorCritic")


def clamp_actions(
    actions: NumpyOrDict, action_space: gymnasium.Space, squash_output: bool
) -> np.ndarray:
    def clip(action: np.ndarray, space: gymnasium.Space) -> np.ndarray:
        if isinstance(space, Box):
            low, high = action_space.low, action_space.high  # type: ignore
            if squash_output:
                # Squashed output is already between -1 and 1; however, low and high
                # might not be -1 and 1.
                return low + 0.5 * (action + 1) * (high - low)
            else:
                return np.clip(action, low, high)
        return action

    if isinstance(actions, dict):
        assert isinstance(action_space, DictSpace)
        clipped = {k: clip(v, action_space[k]) for k, v in actions.items()}  # type: ignore

        return np.array(
            [
                {k: clipped[k][i] for k in clipped}
                for i in range(len(next(iter(clipped.values()))))
            ]
        )

    return clip(actions, action_space)


class OnPolicy(Policy, Generic[ObsType]):
    @property
    @abstractmethod
    def action_shape(self) -> ActionShape: ...

    @property
    def value_shape(self) -> Tuple[int, ...]:
        return ()


class ActorCritic(OnPolicy, Generic[ObsType]):
    def __init__(
        self,
        env_spaces: EnvSpaces,
        pi_hidden_sizes: Optional[Sequence[int]] = None,
        v_hidden_sizes: Optional[Sequence[int]] = None,
        init_layers_orthogonal: bool = True,
        activation_fn: str = "tanh",
        log_std_init: float = -0.5,
        use_sde: bool = False,
        full_std: bool = True,
        squash_output: bool = False,
        share_features_extractor: bool = True,
        cnn_flatten_dim: int = 512,
        cnn_style: str = "nature",
        cnn_layers_init_orthogonal: Optional[bool] = None,
        impala_channels: Sequence[int] = (16, 32, 32),
        actor_head_style: str = "single",
        embed_layer: bool = False,
        backbone_channels: int = 128,
        pooled_channels: int = 512,
        critic_channels: int = 64,
        in_num_res_blocks: int = 4,
        cone_num_res_blocks: int = 6,
        out_num_res_blocks: int = 4,
        num_additional_critics: int = 0,
        additional_critic_activation_functions: Optional[List[str]] = None,
        gelu_pool_conv: bool = True,
        channels_per_level: Optional[List[int]] = None,
        strides_per_level: Optional[List[Union[int, List[int]]]] = None,
        deconv_strides_per_level: Optional[List[Union[int, List[int]]]] = None,
        encoder_residual_blocks_per_level: Optional[List[int]] = None,
        decoder_residual_blocks_per_level: Optional[List[int]] = None,
        increment_kernel_size_on_down_conv: bool = False,
        output_activation_fn: str = "identity",
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
        critic_shares_backbone: Optional[bool] = None,
        save_critic_separate: Optional[bool] = None,
        shared_critic_head: Optional[bool] = None,
        critic_avg_max_pool: bool = False,
        normalization: Optional[str] = None,
        post_backbone_normalization: Optional[str] = None,
        critic_strides: Optional[List[Union[int, List[int]]]] = None,
        encoder_embed_dim: int = 64,
        encoder_attention_heads: int = 4,
        encoder_feed_forward_dim: int = 256,
        encoder_layers: int = 4,
        add_position_features: bool = True,
        actor_head_kernel_size: int = 1,
        key_mask_empty_spaces: bool = True,
        identity_map_reordering: bool = True,
        filter_key_mask_empty_spaces: bool = True,
        hidden_critic_dims: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(env_spaces, **kwargs)
        (
            single_observation_space,
            single_action_space,
            action_plane_space,
            _,  # action_shape,
            _,  # num_envs,
            _,  # reward_shape,
        ) = env_spaces

        self.squash_output = squash_output

        if actor_head_style == "unet":
            assert action_plane_space is not None
            self.network = UNetActorCriticNetwork(
                single_observation_space,
                single_action_space,
                action_plane_space,
                v_hidden_sizes=v_hidden_sizes,
                init_layers_orthogonal=init_layers_orthogonal,
                activation_fn=activation_fn,
                cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
                embed_layer=embed_layer,
            )
        elif actor_head_style == "double_cone":
            assert action_plane_space is not None
            self.network = DoubleConeActorCritic(
                single_observation_space,
                single_action_space,
                action_plane_space,
                init_layers_orthogonal=init_layers_orthogonal,
                cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
                backbone_channels=backbone_channels,
                pooled_channels=pooled_channels,
                critic_channels=critic_channels,
                in_num_res_blocks=in_num_res_blocks,
                cone_num_res_blocks=cone_num_res_blocks,
                out_num_res_blocks=out_num_res_blocks,
                num_additional_critics=num_additional_critics,
                additional_critic_activation_functions=additional_critic_activation_functions,
                output_activation_fn=output_activation_fn,
                gelu_pool_conv=gelu_pool_conv,
                subaction_mask=subaction_mask,
            )
        elif actor_head_style == "squeeze_unet":
            assert action_plane_space is not None
            self.network = SqueezeUnetActorCriticNetwork(
                single_observation_space,
                single_action_space,
                action_plane_space,
                init_layers_orthogonal=init_layers_orthogonal,
                cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
                num_additional_critics=num_additional_critics,
                additional_critic_activation_functions=additional_critic_activation_functions,
                critic_channels=critic_channels,
                channels_per_level=channels_per_level,
                strides_per_level=strides_per_level,
                deconv_strides_per_level=deconv_strides_per_level,
                encoder_residual_blocks_per_level=encoder_residual_blocks_per_level,
                decoder_residual_blocks_per_level=decoder_residual_blocks_per_level,
                increment_kernel_size_on_down_conv=increment_kernel_size_on_down_conv,
                output_activation_fn=output_activation_fn,
                subaction_mask=subaction_mask,
                critic_shares_backbone=(
                    critic_shares_backbone
                    if critic_shares_backbone is not None
                    else True
                ),
                save_critic_separate=(
                    save_critic_separate if save_critic_separate is not None else False
                ),
                shared_critic_head=(
                    shared_critic_head if shared_critic_head is not None else False
                ),
                normalization=normalization,
            )
        elif actor_head_style == "sacus":
            assert action_plane_space is not None
            self.network = SplitActorCriticUShapedNetwork(
                single_observation_space,
                single_action_space,
                action_plane_space,
                init_layers_orthogonal=init_layers_orthogonal,
                cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
                num_additional_critics=num_additional_critics,
                additional_critic_activation_functions=additional_critic_activation_functions,
                critic_channels=critic_channels,
                channels_per_level=channels_per_level,
                strides_per_level=strides_per_level,
                deconv_strides_per_level=deconv_strides_per_level,
                encoder_residual_blocks_per_level=encoder_residual_blocks_per_level,
                decoder_residual_blocks_per_level=decoder_residual_blocks_per_level,
                increment_kernel_size_on_down_conv=increment_kernel_size_on_down_conv,
                output_activation_fn=output_activation_fn,
                subaction_mask=subaction_mask,
                shared_critic_head=(
                    shared_critic_head if shared_critic_head is not None else False
                ),
                critic_avg_max_pool=critic_avg_max_pool,
            )
        elif actor_head_style == "grid2seq_transformer":
            assert action_plane_space is not None
            assert normalization is not None
            self.network = Grid2SeqTransformerNetwork(
                single_observation_space,
                single_action_space,
                action_plane_space,
                init_layers_orthogonal=init_layers_orthogonal,
                cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
                encoder_embed_dim=encoder_embed_dim,
                encoder_attention_heads=encoder_attention_heads,
                encoder_feed_forward_dim=encoder_feed_forward_dim,
                encoder_layers=encoder_layers,
                num_additional_critics=num_additional_critics,
                additional_critic_activation_functions=additional_critic_activation_functions,
                critic_channels=critic_channels,
                critic_strides=critic_strides,
                output_activation_fn=output_activation_fn,
                subaction_mask=subaction_mask,
                shared_critic_head=(
                    shared_critic_head if shared_critic_head is not None else False
                ),
                normalization=normalization,
                add_position_features=add_position_features,
                actor_head_kernel_size=actor_head_kernel_size,
                key_mask_empty_spaces=key_mask_empty_spaces,
                identity_map_reordering=identity_map_reordering,
                filter_key_mask_empty_spaces=filter_key_mask_empty_spaces,
            )
        elif actor_head_style == "grid2entity_transformer":
            assert action_plane_space is not None
            assert normalization is not None
            assert post_backbone_normalization is not None
            self.network = Grid2EntityTransformerNetwork(
                single_observation_space,
                single_action_space,
                action_plane_space,
                init_layers_orthogonal=init_layers_orthogonal,
                encoder_embed_dim=encoder_embed_dim,
                encoder_attention_heads=encoder_attention_heads,
                encoder_feed_forward_dim=encoder_feed_forward_dim,
                encoder_layers=encoder_layers,
                num_additional_critics=num_additional_critics,
                additional_critic_activation_functions=additional_critic_activation_functions,
                hidden_critic_dims=hidden_critic_dims,
                output_activation_fn=output_activation_fn,
                subaction_mask=subaction_mask,
                normalization=normalization,
                post_backbone_normalization=post_backbone_normalization,
                add_position_features=add_position_features,
            )
        elif share_features_extractor:
            self.network = ConnectedTrioActorCriticNetwork(
                single_observation_space,
                single_action_space,
                pi_hidden_sizes=pi_hidden_sizes,
                v_hidden_sizes=v_hidden_sizes,
                init_layers_orthogonal=init_layers_orthogonal,
                activation_fn=activation_fn,
                log_std_init=log_std_init,
                use_sde=use_sde,
                full_std=full_std,
                squash_output=squash_output,
                cnn_flatten_dim=cnn_flatten_dim,
                cnn_style=cnn_style,
                cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
                impala_channels=impala_channels,
                actor_head_style=actor_head_style,
                action_plane_space=action_plane_space,
            )
        else:
            self.network = SeparateActorCriticNetwork(
                single_observation_space,
                single_action_space,
                pi_hidden_sizes=pi_hidden_sizes,
                v_hidden_sizes=v_hidden_sizes,
                init_layers_orthogonal=init_layers_orthogonal,
                activation_fn=activation_fn,
                log_std_init=log_std_init,
                use_sde=use_sde,
                full_std=full_std,
                squash_output=squash_output,
                cnn_flatten_dim=cnn_flatten_dim,
                cnn_style=cnn_style,
                cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
                impala_channels=impala_channels,
                actor_head_style=actor_head_style,
                action_plane_space=action_plane_space,
            )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> ACForward:
        (_, logp_a, entropy), v = self.network(obs, action, action_masks=action_masks)

        assert logp_a is not None
        assert entropy is not None
        return ACForward(logp_a, entropy, v)

    def value(self, obs: ObsType) -> np.ndarray:
        assert isinstance(obs, np.ndarray)
        o = self._as_tensor(obs)
        assert isinstance(o, torch.Tensor)
        with torch.no_grad():
            v = self.network.value(o)
        return v.cpu().numpy()

    def step(self, obs: ObsType, action_masks: Optional[NumpyOrDict] = None) -> Step:
        assert isinstance(obs, np.ndarray)
        o = self._as_tensor(obs)
        assert isinstance(o, torch.Tensor)
        a_masks = self._as_tensor(action_masks) if action_masks is not None else None
        with torch.no_grad():
            (pi, _, _), v = self.network.distribution_and_value(o, action_masks=a_masks)
            a = pi.sample()
            logp_a = pi.log_prob(a)

        a_np = tensor_to_numpy(a)
        clamped_a_np = clamp_actions(
            a_np, self.env_spaces.single_action_space, self.squash_output
        )
        return Step(a_np, v.cpu().numpy(), logp_a.cpu().numpy(), clamped_a_np)

    def logprobs(
        self,
        obs: np.ndarray,
        actions: NumpyOrDict,
        action_masks: Optional[NumpyOrDict] = None,
    ) -> np.ndarray:
        o = self._as_tensor(obs)
        a = self._as_tensor(actions)
        a_masks = self._as_tensor(action_masks) if action_masks is not None else None
        with torch.no_grad():
            (_, logp_a, _), _ = self.network(o, a, action_masks=a_masks)
        return logp_a.cpu().numpy()

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        action_masks: Optional[NumpyOrDict] = None,
    ) -> np.ndarray:
        if not deterministic:
            return self.step(obs, action_masks=action_masks).clamped_a
        else:
            o = self._as_tensor(obs)
            a_masks = (
                self._as_tensor(action_masks) if action_masks is not None else None
            )
            with torch.no_grad():
                (pi, _, _), _ = self.network.distribution_and_value(
                    o, action_masks=a_masks
                )
                a = pi.mode
            return clamp_actions(
                tensor_to_numpy(a),
                self.env_spaces.single_action_space,
                self.squash_output,
            )

    def get_state(self) -> Any:
        if (
            isinstance(self.network, BackboneActorCritic)
            and self.network.save_critic_separate
        ):
            return self.network.get_state()
        else:
            return super().get_state()

    def load_state(self, state: Any) -> None:
        if (
            isinstance(self.network, BackboneActorCritic)
            and self.network.save_critic_separate
        ):
            self.network.set_state(state)
        else:
            super().load_state(state)
        self.reset_noise()

    def save(self, path: str) -> None:
        if (
            isinstance(self.network, BackboneActorCritic)
            and self.network.save_critic_separate
        ):
            self.network.save(path)
        else:
            super().save(path)

    def load(self, path: str) -> None:
        if (
            isinstance(self.network, BackboneActorCritic)
            and self.network.save_critic_separate
        ):
            self.network.load(path, self.device)
        else:
            super().load(path)
        self.reset_noise()

    def reset_noise(self, batch_size: Optional[int] = None) -> None:
        self.network.reset_noise(
            batch_size=batch_size if batch_size else self.env_spaces.num_envs
        )

    @property
    def action_shape(self) -> ActionShape:
        return self.env_spaces.action_shape

    @property
    def value_shape(self) -> Tuple[int, ...]:
        return self.env_spaces.reward_shape

    def freeze(
        self,
        freeze_policy_head: bool,
        freeze_value_head: bool,
        freeze_backbone: bool = True,
    ) -> None:
        self.network.freeze(
            freeze_policy_head, freeze_value_head, freeze_backbone=freeze_backbone
        )

    def unfreeze(self) -> None:
        self.network.unfreeze()
