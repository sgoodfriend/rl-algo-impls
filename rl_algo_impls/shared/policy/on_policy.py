import gym
import numpy as np
import torch

from abc import abstractmethod
from gym.spaces import Box, Discrete, Space
from typing import NamedTuple, Optional, Sequence, Tuple, TypeVar

from rl_algo_impls.shared.module.feature_extractor import FeatureExtractor
from rl_algo_impls.shared.policy.actor import (
    PiForward,
    StateDependentNoiseActorHead,
    actor_head,
)
from rl_algo_impls.shared.policy.critic import CriticHead
from rl_algo_impls.shared.policy.policy import ACTIVATION, Policy
from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnv,
    VecEnvObs,
    single_observation_space,
    single_action_space,
)


class Step(NamedTuple):
    a: np.ndarray
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
    actions: np.ndarray, action_space: gym.Space, squash_output: bool
) -> np.ndarray:
    if isinstance(action_space, Box):
        low, high = action_space.low, action_space.high  # type: ignore
        if squash_output:
            # Squashed output is already between -1 and 1. Rescale if the actual
            # output needs to something other than -1 and 1
            return low + 0.5 * (actions + 1) * (high - low)
        else:
            return np.clip(actions, low, high)
    return actions


def default_hidden_sizes(obs_space: Space) -> Sequence[int]:
    if isinstance(obs_space, Box):
        if len(obs_space.shape) == 3:
            # By default feature extractor to output has no hidden layers
            return []
        elif len(obs_space.shape) == 1:
            return [64, 64]
        else:
            raise ValueError(f"Unsupported observation space: {obs_space}")
    elif isinstance(obs_space, Discrete):
        return [64]
    else:
        raise ValueError(f"Unsupported observation space: {obs_space}")


class OnPolicy(Policy):
    @abstractmethod
    def value(self, obs: VecEnvObs) -> np.ndarray:
        ...

    @abstractmethod
    def step(self, obs: VecEnvObs) -> Step:
        ...


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
        cnn_feature_dim: int = 512,
        cnn_style: str = "nature",
        cnn_layers_init_orthogonal: Optional[bool] = None,
        impala_channels: Sequence[int] = (16, 32, 32),
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)

        observation_space = single_observation_space(env)
        action_space = single_action_space(env)

        pi_hidden_sizes = (
            pi_hidden_sizes
            if pi_hidden_sizes is not None
            else default_hidden_sizes(observation_space)
        )
        v_hidden_sizes = (
            v_hidden_sizes
            if v_hidden_sizes is not None
            else default_hidden_sizes(observation_space)
        )

        activation = ACTIVATION[activation_fn]
        self.action_space = action_space
        self.squash_output = squash_output
        self.share_features_extractor = share_features_extractor
        self._feature_extractor = FeatureExtractor(
            observation_space,
            activation,
            init_layers_orthogonal=init_layers_orthogonal,
            cnn_feature_dim=cnn_feature_dim,
            cnn_style=cnn_style,
            cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
            impala_channels=impala_channels,
        )
        self._pi = actor_head(
            self.action_space,
            (self._feature_extractor.out_dim,) + tuple(pi_hidden_sizes),
            init_layers_orthogonal,
            activation,
            log_std_init=log_std_init,
            use_sde=use_sde,
            full_std=full_std,
            squash_output=squash_output,
        )

        if not share_features_extractor:
            self._v_feature_extractor = FeatureExtractor(
                observation_space,
                activation,
                init_layers_orthogonal=init_layers_orthogonal,
                cnn_feature_dim=cnn_feature_dim,
                cnn_style=cnn_style,
                cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
            )
            v_hidden_sizes = (self._v_feature_extractor.out_dim,) + tuple(
                v_hidden_sizes
            )
        else:
            self._v_feature_extractor = None
            v_hidden_sizes = (self._feature_extractor.out_dim,) + tuple(v_hidden_sizes)
        self._v = CriticHead(
            hidden_sizes=v_hidden_sizes,
            activation=activation,
            init_layers_orthogonal=init_layers_orthogonal,
        )

    def _pi_forward(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[PiForward, torch.Tensor]:
        p_fe = self._feature_extractor(obs)
        pi_forward = self._pi(p_fe, action)

        return pi_forward, p_fe

    def _v_forward(self, obs: torch.Tensor, p_fc: torch.Tensor) -> torch.Tensor:
        v_fe = self._v_feature_extractor(obs) if self._v_feature_extractor else p_fc
        return self._v(v_fe)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> ACForward:
        (_, logp_a, entropy), p_fc = self._pi_forward(obs, action)
        v = self._v_forward(obs, p_fc)

        assert logp_a is not None
        assert entropy is not None
        return ACForward(logp_a, entropy, v)

    def value(self, obs: VecEnvObs) -> np.ndarray:
        o = self._as_tensor(obs)
        with torch.no_grad():
            fe = (
                self._v_feature_extractor(o)
                if self._v_feature_extractor
                else self._feature_extractor(o)
            )
            v = self._v(fe)
        return v.cpu().numpy()

    def step(self, obs: VecEnvObs) -> Step:
        o = self._as_tensor(obs)
        with torch.no_grad():
            (pi, _, _), p_fc = self._pi_forward(o)
            a = pi.sample()
            logp_a = pi.log_prob(a)

            v = self._v_forward(o, p_fc)

        a_np = a.cpu().numpy()
        clamped_a_np = clamp_actions(a_np, self.action_space, self.squash_output)
        return Step(a_np, v.cpu().numpy(), logp_a.cpu().numpy(), clamped_a_np)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if not deterministic:
            return self.step(obs).clamped_a
        else:
            o = self._as_tensor(obs)
            with torch.no_grad():
                (pi, _, _), _ = self._pi_forward(o)
                a = pi.mode
            return clamp_actions(a.cpu().numpy(), self.action_space, self.squash_output)

    def load(self, path: str) -> None:
        super().load(path)
        self.reset_noise()

    def reset_noise(self, batch_size: Optional[int] = None) -> None:
        if isinstance(self._pi, StateDependentNoiseActorHead):
            self._pi.sample_weights(
                batch_size=batch_size if batch_size else self.env.num_envs
            )
