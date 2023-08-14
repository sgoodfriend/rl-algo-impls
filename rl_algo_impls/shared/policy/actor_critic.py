from abc import abstractmethod
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, TypeVar, Union

import gym
import numpy as np
import torch
from gym.spaces import Box
from gym.spaces import Dict as DictSpace

from rl_algo_impls.shared.policy.actor_critic_network import (
    ConnectedTrioActorCriticNetwork,
    SeparateActorCriticNetwork,
    UNetActorCriticNetwork,
)
from rl_algo_impls.shared.policy.actor_critic_network.double_cone import (
    DoubleConeActorCritic,
)
from rl_algo_impls.shared.policy.actor_critic_network.squeeze_unet import (
    SqueezeUnetActorCriticNetwork,
)
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.tensor_utils import NumpyOrDict, tensor_to_numpy
from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnv,
    VecEnvObs,
    single_action_space,
    single_observation_space,
)


class Step(NamedTuple):
    a: NumpyOrDict
    v: np.ndarray
    logp_a: np.ndarray
    clamped_a: np.ndarray


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
    actions: NumpyOrDict, action_space: gym.Space, squash_output: bool
) -> np.ndarray:
    def clip(action: np.ndarray, space: gym.Space) -> np.ndarray:
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


class OnPolicy(Policy):
    @abstractmethod
    def value(self, obs: VecEnvObs) -> np.ndarray:
        ...

    @abstractmethod
    def step(self, obs: VecEnvObs, action_masks: Optional[NumpyOrDict] = None) -> Step:
        ...

    @property
    @abstractmethod
    def action_shape(self) -> Tuple[int, ...]:
        ...

    @property
    def value_shape(self) -> Tuple[int, ...]:
        return ()


class ActorCritic(OnPolicy):
    def __init__(
        self,
        env: VecEnv,
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
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)

        observation_space = single_observation_space(env)
        action_space = single_action_space(env)
        action_plane_space = getattr(env, "action_plane_space", None)

        self.action_space = action_space
        self.squash_output = squash_output

        if actor_head_style == "unet":
            assert action_plane_space is not None
            self.network = UNetActorCriticNetwork(
                observation_space,
                action_space,
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
                observation_space,
                action_space,
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
                observation_space,
                action_space,
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
            )
        elif share_features_extractor:
            self.network = ConnectedTrioActorCriticNetwork(
                observation_space,
                action_space,
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
                observation_space,
                action_space,
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

    def value(self, obs: VecEnvObs) -> np.ndarray:
        assert isinstance(obs, np.ndarray)
        o = self._as_tensor(obs)
        assert isinstance(o, torch.Tensor)
        with torch.no_grad():
            v = self.network.value(o)
        return v.cpu().numpy()

    def step(self, obs: VecEnvObs, action_masks: Optional[NumpyOrDict] = None) -> Step:
        assert isinstance(obs, np.ndarray)
        o = self._as_tensor(obs)
        assert isinstance(o, torch.Tensor)
        a_masks = self._as_tensor(action_masks) if action_masks is not None else None
        with torch.no_grad():
            (pi, _, _), v = self.network.distribution_and_value(o, action_masks=a_masks)
            a = pi.sample()
            logp_a = pi.log_prob(a)

        a_np = tensor_to_numpy(a)
        clamped_a_np = clamp_actions(a_np, self.action_space, self.squash_output)
        return Step(a_np, v.cpu().numpy(), logp_a.cpu().numpy(), clamped_a_np)

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
                tensor_to_numpy(a), self.action_space, self.squash_output
            )

    def load(self, path: str) -> None:
        super().load(path)
        self.reset_noise()

    def load_from(self: ActorCriticSelf, policy: ActorCriticSelf) -> ActorCriticSelf:
        super().load_from(policy)
        self.reset_noise()
        return self

    def reset_noise(self, batch_size: Optional[int] = None) -> None:
        self.network.reset_noise(
            batch_size=batch_size if batch_size else self.env.num_envs
        )

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return self.network.action_shape

    @property
    def value_shape(self) -> Tuple[int, ...]:
        return self.network.value_shape
