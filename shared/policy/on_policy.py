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
        obs_space: gym.Space,
        act_dim: int,
        hidden_sizes: Sequence[int] = (32,),
        activation: Type[nn.Module] = nn.Tanh,
        init_layers_orthogonal: bool = True,
    ) -> None:
        super().__init__()
        layer_sizes = tuple(hidden_sizes) + (act_dim,)
        self._preprocessor, self._feature_extractor = feature_extractor(
            obs_space,
            activation,
            layer_sizes[0],
            init_layers_orthogonal=init_layers_orthogonal,
        )
        self._fc = mlp(
            layer_sizes,
            activation,
            init_layers_orthogonal=init_layers_orthogonal,
            final_layer_gain=0.01,
        )

    def forward(self, obs: torch.Tensor, a: Optional[torch.Tensor] = None) -> PiForward:
        obs = self._preprocessor(obs) if self._preprocessor else obs
        obs = self._feature_extractor(obs)
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
        obs_space: gym.Space,
        act_dim: int,
        hidden_sizes: Sequence[int] = (32,),
        activation: Type[nn.Module] = nn.Tanh,
        init_layers_orthogonal: bool = True,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        layer_sizes = tuple(hidden_sizes) + (act_dim,)
        self._preprocessor, self._feature_extractor = feature_extractor(
            obs_space,
            activation,
            layer_sizes[0],
            init_layers_orthogonal=init_layers_orthogonal,
        )
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
        obs = self._preprocessor(obs) if self._preprocessor else obs
        obs = self._feature_extractor(obs)
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
        obs_space: gym.Space,
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
        self._preprocessor, self._feature_extractor = feature_extractor(
            obs_space,
            activation,
            layer_sizes[0],
            init_layers_orthogonal=init_layers_orthogonal,
        )
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
        obs = self._preprocessor(obs) if self._preprocessor else obs
        obs = self._feature_extractor(obs)
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
        obs_space: gym.Space,
        hidden_sizes: Sequence[int] = (32,),
        activation: Type[nn.Module] = nn.Tanh,
        init_layers_orthogonal: bool = True,
    ) -> None:
        super().__init__()
        layer_sizes = tuple(hidden_sizes) + (1,)
        self._preprocessor, self._feature_extractor = feature_extractor(
            obs_space,
            activation,
            layer_sizes[0],
            init_layers_orthogonal=init_layers_orthogonal,
        )
        self._fc = mlp(
            layer_sizes,
            activation,
            init_layers_orthogonal=init_layers_orthogonal,
            final_layer_gain=1.0,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self._preprocessor(obs) if self._preprocessor else obs
        obs = self._feature_extractor(obs)
        v = self._fc(obs)
        return v.squeeze(-1)


class Step(NamedTuple):
    a: np.ndarray
    v: np.ndarray
    logp_a: np.ndarray


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
        **kwargs,
    ) -> None:
        super().__init__(env)
        activation = ACTIVATION[activation_fn]
        observation_space = env.observation_space
        action_space = env.action_space
        if isinstance(action_space, Discrete):
            self.pi = CategoricalActor(
                observation_space,
                action_space.n,
                hidden_sizes=pi_hidden_sizes,
                activation=activation,
                init_layers_orthogonal=init_layers_orthogonal,
            )
        elif isinstance(action_space, Box):
            if use_sde:
                self.pi = StateDependentNoiseActor(
                    observation_space,
                    action_space.shape[0],
                    hidden_sizes=pi_hidden_sizes,
                    activation=activation,
                    init_layers_orthogonal=init_layers_orthogonal,
                    log_std_init=log_std_init,
                    full_std=full_std,
                    squash_output=squash_output,
                )
            else:
                self.pi = GaussianActor(
                    observation_space,
                    action_space.shape[0],
                    hidden_sizes=pi_hidden_sizes,
                    activation=activation,
                    init_layers_orthogonal=init_layers_orthogonal,
                    log_std_init=log_std_init,
                )
        else:
            raise ValueError(f"Unsupported action space: {action_space}")
        self.pi.train(self.training)

        self.v = Critic(
            observation_space,
            hidden_sizes=v_hidden_sizes,
            activation=activation,
            init_layers_orthogonal=init_layers_orthogonal,
        ).train(self.training)

    def step(self, obs: VecEnvObs) -> Step:
        assert isinstance(obs, np.ndarray)
        o = torch.as_tensor(obs)
        if self.device is not None:
            o = o.to(self.device)
        with torch.no_grad():
            pi, _, _ = self.pi(o)
            a = pi.sample()
            v = self.v(o)
            logp_a = pi.log_prob(a)
        return Step(a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy())

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if not deterministic:
            return self.step(obs).a
        else:
            assert isinstance(obs, np.ndarray)
            o = torch.as_tensor(obs)
            if self.device is not None:
                o = o.to(self.device)
            with torch.no_grad():
                pi, _, _ = self.pi(o)
                a = pi.mode
            return a.cpu().numpy()

    def save(self, path: str) -> None:
        super().save(path)
        torch.save(self.pi.state_dict(), Path(path) / "pi.pt")
        torch.save(self.v.state_dict(), Path(path) / "v.pt")

    def load(self, path: str) -> None:
        self.pi.load_state_dict(torch.load(Path(path) / "pi.pt"))
        self.v.load_state_dict(torch.load(Path(path) / "v.pt"))
        self.reset_noise()

    def reset_noise(self, batch_size: Optional[int] = None) -> None:
        if isinstance(self.pi, StateDependentNoiseActor):
            self.pi.sample_weights(
                batch_size=batch_size if batch_size else self.env.num_envs
            )
