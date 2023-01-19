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
from typing import NamedTuple, Optional, Sequence, Type, TypeVar

from shared.module import feature_extractor, mlp
from shared.policy.policy import ACTIVATION, Policy


class PiForward(NamedTuple):
    pi: Distribution
    logp_a: Optional[torch.Tensor]


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
        if a is not None:
            logp_a = pi.log_prob(a)
        return PiForward(pi, logp_a)


class GaussianActor(Actor):
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
        self.mu_net = mlp(
            layer_sizes,
            activation,
            init_layers_orthogonal=init_layers_orthogonal,
            final_layer_gain=0.01,
        )
        self.log_std = nn.Parameter(torch.ones(act_dim, dtype=torch.float32) * -0.5)

    def _distribution(self, obs: torch.Tensor) -> Distribution:
        obs = self._preprocessor(obs) if self._preprocessor else obs
        obs = self._feature_extractor(obs)
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distrubition(
        self, pi: Distribution, a: torch.Tensor
    ) -> torch.Tensor:
        return pi.log_prob(a).sum(axis=-1)

    def forward(self, obs: torch.Tensor, a: Optional[torch.Tensor] = None) -> PiForward:
        pi = self._distribution(obs)
        logp_a = None
        if a is not None:
            logp_a = self._log_prob_from_distrubition(pi, a)
        return PiForward(pi, logp_a)


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
            ).train(self.training)
        elif isinstance(action_space, Box):
            self.pi = GaussianActor(
                observation_space,
                action_space.shape[0],
                hidden_sizes=pi_hidden_sizes,
                activation=activation,
                init_layers_orthogonal=init_layers_orthogonal,
            ).train(self.training)
        else:
            raise ValueError(f"Unsupported action space: {action_space}")

        self.v = Critic(
            observation_space,
            hidden_sizes=v_hidden_sizes,
            activation=activation,
            init_layers_orthogonal=init_layers_orthogonal,
        ).train(self.training)

    def step(self, obs: VecEnvObs) -> Step:
        assert isinstance(obs, np.ndarray)
        o = torch.as_tensor(np.array(obs))
        if self.device is not None:
            o = o.to(self.device)
        with torch.no_grad():
            pi = self.pi(o)
            a = pi.pi.sample()
            v = self.v(o)
            logp_a = pi.pi.log_prob(a)
        return Step(a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy())

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.step(obs).a

    def save(self, path: str) -> None:
        super().save(path)
        torch.save(self.pi.state_dict(), Path(path) / "pi.pt")
        torch.save(self.v.state_dict(), Path(path) / "v.pt")

    def load(self, path: str) -> None:
        self.pi.load_state_dict(torch.load(Path(path) / "pi.pt"))
        self.v.load_state_dict(torch.load(Path(path) / "v.pt"))
