import gym
import numpy as np
import os
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from gym.spaces import Box, Discrete
from pathlib import Path
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs
from torch.distributions import Categorical, Distribution, Normal
from typing import NamedTuple, Optional, Sequence, Type, TypeVar, Union

from shared.module import feature_extractor, mlp
from shared.policy.policy import ACTIVATION, Policy


class PiForward(NamedTuple):
    pi: Distribution
    logp_a: Optional[torch.Tensor]
    entropy: Optional[torch.Tensor]


class Actor(nn.Module, ABC):
    @abstractmethod
    def forward(self, obs: torch.Tensor, a: Optional[torch.Tensor] = None) -> PiForward:
        ...


class CategoricalActor(Actor):
    def __init__(
        self,
        act_dim: int,
        hidden_sizes: Sequence[int] = (32,),
        activation: Type[nn.Module] = nn.Tanh,
        init_layers_orthogonal: bool = True,
    ) -> None:
        super().__init__()
        layer_sizes = tuple(hidden_sizes) + (act_dim,)
        self._fc = mlp(
            layer_sizes,
            activation,
            init_layers_orthogonal=init_layers_orthogonal,
            final_layer_gain=0.01,
        )

    def forward(self, obs: torch.Tensor, a: Optional[torch.Tensor] = None) -> PiForward:
        logits = self._fc(obs)
        pi = Categorical(logits=logits)
        logp_a = None
        entropy = None
        if a is not None:
            logp_a = pi.log_prob(a)
            entropy = pi.entropy()
        return PiForward(pi, logp_a, entropy)


class GaussianDistribution(Normal):
    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        return super().log_prob(a).sum(axis=-1)

    def sample(self) -> torch.Tensor:
        return self.rsample()


class GaussianActor(Actor):
    def __init__(
        self,
        act_dim: int,
        hidden_sizes: Sequence[int] = (32,),
        activation: Type[nn.Module] = nn.Tanh,
        init_layers_orthogonal: bool = True,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        layer_sizes = tuple(hidden_sizes) + (act_dim,)
        self.mu_net = mlp(
            layer_sizes,
            activation,
            init_layers_orthogonal=init_layers_orthogonal,
            final_layer_gain=0.01,
        )
        self.log_std = nn.Parameter(
            torch.ones(act_dim, dtype=torch.float32) * log_std_init
        )

    def _distribution(self, obs: torch.Tensor) -> Distribution:
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return GaussianDistribution(mu, std)

    def forward(self, obs: torch.Tensor, a: Optional[torch.Tensor] = None) -> PiForward:
        pi = self._distribution(obs)
        logp_a = None
        entropy = None
        if a is not None:
            logp_a = pi.log_prob(a)
            entropy = pi.entropy()
        return PiForward(pi, logp_a, entropy)


class TanhBijector:
    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def inverse(y: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(y.dtype).eps
        clamped_y = y.clamp(min=-1.0 + eps, max=1.0 - eps)
        return torch.atanh(clamped_y)

    def log_prob_correction(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(1.0 - torch.tanh(x) ** 2 + self.epsilon)


class StateDependentNoiseDistribution(Normal):
    def __init__(
        self,
        loc,
        scale,
        latent_sde: torch.Tensor,
        exploration_mat: torch.Tensor,
        exploration_matrices: torch.Tensor,
        bijector: Optional[TanhBijector] = None,
        validate_args=None,
    ):
        super().__init__(loc, scale, validate_args)
        self.latent_sde = latent_sde
        self.exploration_mat = exploration_mat
        self.exploration_matrices = exploration_matrices
        self.bijector = bijector

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        gaussian_a = self.bijector.inverse(a) if self.bijector else a
        log_prob = super().log_prob(gaussian_a).sum(axis=-1)
        if self.bijector:
            log_prob -= torch.sum(self.bijector.log_prob_correction(gaussian_a), dim=1)
        return log_prob

    def sample(self) -> torch.Tensor:
        noise = self._get_noise()
        actions = self.mean + noise
        return self.bijector.forward(actions) if self.bijector else actions

    def _get_noise(self) -> torch.Tensor:
        if len(self.latent_sde) == 1 or len(self.latent_sde) != len(
            self.exploration_matrices
        ):
            return torch.mm(self.latent_sde, self.exploration_mat)
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = self.latent_sde.unsqueeze(dim=1)
        # (batch_size, 1, n_actions)
        noise = torch.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(dim=1)

    @property
    def mode(self) -> torch.Tensor:
        mean = super().mode
        return self.bijector.forward(mean) if self.bijector else mean


StateDependentNoiseActorSelf = TypeVar(
    "StateDependentNoiseActorSelf", bound="StateDependentNoiseActor"
)


class StateDependentNoiseActor(Actor):
    def __init__(
        self,
        act_dim: int,
        hidden_sizes: Sequence[int] = (32,),
        activation: Type[nn.Module] = nn.Tanh,
        init_layers_orthogonal: bool = True,
        log_std_init: float = -0.5,
        full_std: bool = True,
        squash_output: bool = False,
        learn_std: bool = False,
    ) -> None:
        super().__init__()
        self.act_dim = act_dim
        layer_sizes = tuple(hidden_sizes) + (self.act_dim,)
        if len(layer_sizes) == 2:
            self.latent_net = nn.Identity()
        elif len(layer_sizes) > 2:
            self.latent_net = mlp(
                layer_sizes[:-1],
                activation,
                output_activation=activation,
                init_layers_orthogonal=init_layers_orthogonal,
            )
        else:
            raise ValueError("hidden_sizes must be of at least length 1")
        self.mu_net = mlp(
            layer_sizes[-2:],
            activation,
            init_layers_orthogonal=init_layers_orthogonal,
            final_layer_gain=0.01,
        )
        self.full_std = full_std
        std_dim = (hidden_sizes[-1], act_dim if self.full_std else 1)
        self.log_std = nn.Parameter(
            torch.ones(std_dim, dtype=torch.float32) * log_std_init
        )
        self.bijector = TanhBijector() if squash_output else None
        self.learn_std = learn_std
        self.device = None

        self.exploration_mat = None
        self.exploration_matrices = None
        self.sample_weights()

    def to(
        self: StateDependentNoiseActorSelf,
        device: Optional[torch.device] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
        non_blocking: bool = False,
    ) -> StateDependentNoiseActorSelf:
        super().to(device, dtype, non_blocking)
        self.device = device
        return self

    def _distribution(self, obs: torch.Tensor) -> Distribution:
        latent = self.latent_net(obs)
        mu = self.mu_net(latent)
        latent_sde = latent if self.learn_std else latent.detach()
        variance = torch.mm(latent_sde**2, self._get_std() ** 2)
        assert self.exploration_mat is not None
        assert self.exploration_matrices is not None
        return StateDependentNoiseDistribution(
            mu,
            torch.sqrt(variance + 1e-6),
            latent_sde,
            self.exploration_mat,
            self.exploration_matrices,
            self.bijector,
        )

    def _get_std(self) -> torch.Tensor:
        std = torch.exp(self.log_std)
        if self.full_std:
            return std
        ones = torch.ones(self.log_std.shape[0], self.act_dim)
        if self.device:
            ones = ones.to(self.device)
        return ones * std

    def forward(self, obs: torch.Tensor, a: Optional[torch.Tensor] = None) -> PiForward:
        pi = self._distribution(obs)
        logp_a = None
        entropy = None
        if a is not None:
            logp_a = pi.log_prob(a)
            entropy = -logp_a
        return PiForward(pi, logp_a, entropy)

    def sample_weights(self, batch_size: int = 1) -> None:
        std = self._get_std()
        weights_dist = Normal(torch.zeros_like(std), std)
        # Reparametrization trick to pass gradients
        self.exploration_mat = weights_dist.rsample()
        self.exploration_matrices = weights_dist.rsample(torch.Size((batch_size,)))


class Critic(nn.Module):
    def __init__(
        self,
        hidden_sizes: Sequence[int] = (32,),
        activation: Type[nn.Module] = nn.Tanh,
        init_layers_orthogonal: bool = True,
    ) -> None:
        super().__init__()
        layer_sizes = tuple(hidden_sizes) + (1,)
        self._fc = mlp(
            layer_sizes,
            activation,
            init_layers_orthogonal=init_layers_orthogonal,
            final_layer_gain=1.0,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        v = self._fc(obs)
        return v.squeeze(-1)


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


class ActorCritic(Policy):
    def __init__(
        self,
        env: VecEnv,
        pi_hidden_sizes: Sequence[int],
        v_hidden_sizes: Sequence[int],
        init_layers_orthogonal: bool = True,
        activation_fn: str = "tanh",
        log_std_init: float = -0.5,
        use_sde: bool = False,
        full_std: bool = True,
        squash_output: bool = False,
        share_features_extractor: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(env)
        activation = ACTIVATION[activation_fn]
        observation_space = env.observation_space
        self.action_space = env.action_space
        self.squash_output = False
        self.share_features_extractor = share_features_extractor
        assert pi_hidden_sizes
        assert v_hidden_sizes
        assert not share_features_extractor or pi_hidden_sizes[0] == v_hidden_sizes[0]
        self._preprocessor, self._feature_extractor = feature_extractor(
            observation_space,
            activation,
            pi_hidden_sizes[0],
            init_layers_orthogonal=init_layers_orthogonal,
        )
        if isinstance(self.action_space, Discrete):
            self._pi = CategoricalActor(
                self.action_space.n,
                hidden_sizes=pi_hidden_sizes,
                activation=activation,
                init_layers_orthogonal=init_layers_orthogonal,
            )
        elif isinstance(self.action_space, Box):
            if use_sde:
                self._pi = StateDependentNoiseActor(
                    self.action_space.shape[0],
                    hidden_sizes=pi_hidden_sizes,
                    activation=activation,
                    init_layers_orthogonal=init_layers_orthogonal,
                    log_std_init=log_std_init,
                    full_std=full_std,
                    squash_output=squash_output,
                )
                self.squash_output = squash_output
            else:
                self._pi = GaussianActor(
                    self.action_space.shape[0],
                    hidden_sizes=pi_hidden_sizes,
                    activation=activation,
                    init_layers_orthogonal=init_layers_orthogonal,
                    log_std_init=log_std_init,
                )
        else:
            raise ValueError(f"Unsupported action space: {self.action_space}")

        self._v_preprocessor, self._v_feature_extractor = None, None
        if not share_features_extractor:
            self._v_preprocessor, self._v_feature_extractor = feature_extractor(
                observation_space,
                activation,
                v_hidden_sizes[0],
                init_layers_orthogonal=init_layers_orthogonal,
            )
        self._v = Critic(
            hidden_sizes=v_hidden_sizes,
            activation=activation,
            init_layers_orthogonal=init_layers_orthogonal,
        )

    def _pi_forward(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> tuple[PiForward, torch.Tensor]:
        if self._preprocessor:
            obs = self._preprocessor(obs)
        p_fc = self._feature_extractor(obs)
        pi_forward = self._pi(p_fc, action)

        return pi_forward, p_fc

    def _v_forward(self, obs: torch.Tensor, p_fc: torch.Tensor) -> torch.Tensor:
        if self._v_preprocessor:
            obs = self._v_preprocessor(obs)
        v_fc = self._v_feature_extractor(obs) if self._v_feature_extractor else p_fc
        return self._v(v_fc)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> ACForward:
        (_, logp_a, entropy), p_fc = self._pi_forward(obs, action)
        v = self._v_forward(obs, p_fc)

        assert logp_a is not None
        assert entropy is not None
        return ACForward(logp_a, entropy, v)

    def _as_tensor(self, obs: VecEnvObs) -> torch.Tensor:
        assert isinstance(obs, np.ndarray)
        o = torch.as_tensor(obs)
        if self.device is not None:
            o = o.to(self.device)
        return o

    def value(self, obs: VecEnvObs) -> np.ndarray:
        o = self._as_tensor(obs)
        with torch.no_grad():
            if self._v_preprocessor:
                o = self._v_preprocessor(o)
            elif self._preprocessor and not self._v_feature_extractor:
                o = self._preprocessor(o)
            fc = (
                self._v_feature_extractor(o)
                if self._v_feature_extractor
                else self._feature_extractor(o)
            )
            v = self._v(fc)
        return v.cpu().numpy()

    def step(self, obs: VecEnvObs) -> Step:
        o = self._as_tensor(obs)
        with torch.no_grad():
            (pi, _, _), p_fc = self._pi_forward(o)
            a = pi.sample()
            logp_a = pi.log_prob(a)

            v = self._v_forward(o, p_fc)

        a_np = a.cpu().numpy()
        clamped_a_np = self._clamp_actions(a_np)
        return Step(a_np, v.cpu().numpy(), logp_a.cpu().numpy(), clamped_a_np)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if not deterministic:
            return self.step(obs).clamped_a
        else:
            o = self._as_tensor(obs)
            with torch.no_grad():
                (pi, _, _), _ = self._pi_forward(o)
                a = pi.mode
            return self._clamp_actions(a.cpu().numpy())

    def save(self, path: str) -> None:
        super().save(path)
        torch.save(
            self._feature_extractor.state_dict(), Path(path) / FEAT_EXT_FILE_NAME
        )
        if self._v_feature_extractor:
            torch.save(
                self._v_feature_extractor.state_dict(),
                Path(path) / V_FEAT_EXT_FILE_NAME,
            )
        torch.save(self._pi.state_dict(), Path(path) / PI_FILE_NAME)
        torch.save(self._v.state_dict(), Path(path) / V_FILE_NAME)

    def load(self, path: str) -> None:
        self._feature_extractor.load_state_dict(
            torch.load(Path(path)) / FEAT_EXT_FILE_NAME
        )
        if self._v_feature_extractor:
            self._v_feature_extractor.load_state_dict(
                torch.load(Path(path) / V_FEAT_EXT_FILE_NAME)
            )
        self._pi.load_state_dict(torch.load(Path(path) / PI_FILE_NAME))
        self._v.load_state_dict(torch.load(Path(path) / V_FILE_NAME))
        self.reset_noise()

    def reset_noise(self, batch_size: Optional[int] = None) -> None:
        if isinstance(self._pi, StateDependentNoiseActor):
            self._pi.sample_weights(
                batch_size=batch_size if batch_size else self.env.num_envs
            )

    def _clamp_actions(self, actions: np.ndarray) -> np.ndarray:
        if isinstance(self.action_space, Box):
            low, high = self.action_space.low, self.action_space.high  # type: ignore
            if self.squash_output:
                # Squashed output is already between -1 and 1. Rescale if the actual
                # output needs to something other than -1 and 1
                return low + 0.5 * (actions + 1) * (high - low)
            else:
                return np.clip(actions, low, high)
        return actions
