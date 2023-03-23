import gym
import numpy as np
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from gym.spaces import Box, Discrete, MultiDiscrete
from numpy.typing import NDArray
from torch.distributions import Categorical, Distribution, Normal, constraints
from typing import Dict, NamedTuple, Optional, Sequence, Type, TypeVar, Union

from rl_algo_impls.shared.module.module import mlp


class PiForward(NamedTuple):
    pi: Distribution
    logp_a: Optional[torch.Tensor]
    entropy: Optional[torch.Tensor]


class Actor(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
    ) -> PiForward:
        ...


class CategoricalActorHead(Actor):
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

    def forward(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
    ) -> PiForward:
        logits = self._fc(obs)
        pi = MaskedCategorical(logits=logits, mask=action_masks)
        logp_a = None
        entropy = None
        if actions is not None:
            logp_a = pi.log_prob(actions)
            entropy = pi.entropy()
        return PiForward(pi, logp_a, entropy)


class MultiCategorical(Distribution):
    def __init__(
        self,
        nvec: NDArray[np.int64],
        probs=None,
        logits=None,
        validate_args=None,
        masks: Optional[torch.Tensor] = None,
    ):
        # Either probs or logits should be set
        assert (probs is None) != (logits is None)
        masks_split = (
            torch.split(masks, nvec.tolist(), dim=1)
            if masks is not None
            else [None] * len(nvec)
        )
        if probs:
            self.dists = [
                MaskedCategorical(probs=p, validate_args=validate_args, mask=m)
                for p, m in zip(torch.split(probs, nvec.tolist(), dim=1), masks_split)
            ]
            param = probs
        else:
            assert logits is not None
            self.dists = [
                MaskedCategorical(logits=lg, validate_args=validate_args, mask=m)
                for lg, m in zip(torch.split(logits, nvec.tolist(), dim=1), masks_split)
            ]
            param = logits
        batch_shape = param.size()[:-1] if param.ndimension() > 1 else torch.Size()
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        prob_stack = torch.stack(
            [c.log_prob(a) for a, c in zip(action.T, self.dists)], dim=-1
        )
        return prob_stack.sum(dim=-1)

    def entropy(self):
        return torch.stack([c.entropy() for c in self.dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape: torch.Size = torch.Size()):
        return torch.stack([c.sample(sample_shape) for c in self.dists], dim=-1)

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        # Constraints handled by child distributions in dist
        return {}


class MaskedCategorical(Categorical):
    def __init__(
        self,
        probs=None,
        logits=None,
        validate_args=None,
        mask: Optional[torch.Tensor] = None,
    ):
        if mask is not None:
            assert logits is not None, "mask requires logits and not probs"
            logits = torch.where(mask, logits, -1e8)
        self.mask = mask
        super().__init__(probs, logits, validate_args)

    def entropy(self) -> torch.Tensor:
        if self.mask is None:
            return super().entropy()
        # If mask set, then use approximation for entropy
        p_log_p = self.logits * self.probs
        masked = torch.where(self.mask, p_log_p, 0)
        return -masked.sum(-1)


class MultiDiscreteActorHead(Actor):
    def __init__(
        self,
        nvec: NDArray[np.int64],
        hidden_sizes: Sequence[int] = (32,),
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: bool = True,
    ) -> None:
        super().__init__()
        self.nvec = nvec
        layer_sizes = tuple(hidden_sizes) + (nvec.sum(),)
        self._fc = mlp(
            layer_sizes,
            activation,
            init_layers_orthogonal=init_layers_orthogonal,
            final_layer_gain=0.01,
        )

    def forward(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
    ) -> PiForward:
        logits = self._fc(obs)
        pi = MultiCategorical(self.nvec, logits=logits, masks=action_masks)
        logp_a = None
        entropy = None
        if actions is not None:
            logp_a = pi.log_prob(actions)
            entropy = pi.entropy()
        return PiForward(pi, logp_a, entropy)


class GaussianDistribution(Normal):
    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        return super().log_prob(a).sum(axis=-1)

    def sample(self) -> torch.Tensor:
        return self.rsample()


class GaussianActorHead(Actor):
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

    def forward(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
    ) -> PiForward:
        assert (
            not action_masks
        ), f"{self.__class__.__name__} does not support action_masks"
        pi = self._distribution(obs)
        logp_a = None
        entropy = None
        if actions is not None:
            logp_a = pi.log_prob(actions)
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


def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) > 1:
        return tensor.sum(dim=1)
    return tensor.sum()


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
        log_prob = sum_independent_dims(super().log_prob(gaussian_a))
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


StateDependentNoiseActorHeadSelf = TypeVar(
    "StateDependentNoiseActorHeadSelf", bound="StateDependentNoiseActorHead"
)


class StateDependentNoiseActorHead(Actor):
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
        self: StateDependentNoiseActorHeadSelf,
        device: Optional[torch.device] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
        non_blocking: bool = False,
    ) -> StateDependentNoiseActorHeadSelf:
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

    def forward(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
    ) -> PiForward:
        assert (
            not action_masks
        ), f"{self.__class__.__name__} does not support action_masks"
        pi = self._distribution(obs)
        logp_a = None
        entropy = None
        if actions is not None:
            logp_a = pi.log_prob(actions)
            entropy = -logp_a if self.bijector else sum_independent_dims(pi.entropy())
        return PiForward(pi, logp_a, entropy)

    def sample_weights(self, batch_size: int = 1) -> None:
        std = self._get_std()
        weights_dist = Normal(torch.zeros_like(std), std)
        # Reparametrization trick to pass gradients
        self.exploration_mat = weights_dist.rsample()
        self.exploration_matrices = weights_dist.rsample(torch.Size((batch_size,)))


def actor_head(
    action_space: gym.Space,
    hidden_sizes: Sequence[int],
    init_layers_orthogonal: bool,
    activation: Type[nn.Module],
    log_std_init: float = -0.5,
    use_sde: bool = False,
    full_std: bool = True,
    squash_output: bool = False,
) -> Actor:
    assert not use_sde or isinstance(
        action_space, Box
    ), "use_sde only valid if Box action_space"
    assert not squash_output or use_sde, "squash_output only valid if use_sde"
    if isinstance(action_space, Discrete):
        return CategoricalActorHead(
            action_space.n,
            hidden_sizes=hidden_sizes,
            activation=activation,
            init_layers_orthogonal=init_layers_orthogonal,
        )
    elif isinstance(action_space, Box):
        if use_sde:
            return StateDependentNoiseActorHead(
                action_space.shape[0],
                hidden_sizes=hidden_sizes,
                activation=activation,
                init_layers_orthogonal=init_layers_orthogonal,
                log_std_init=log_std_init,
                full_std=full_std,
                squash_output=squash_output,
            )
        else:
            return GaussianActorHead(
                action_space.shape[0],
                hidden_sizes=hidden_sizes,
                activation=activation,
                init_layers_orthogonal=init_layers_orthogonal,
                log_std_init=log_std_init,
            )
    elif isinstance(action_space, MultiDiscrete):
        return MultiDiscreteActorHead(
            action_space.nvec,
            hidden_sizes=hidden_sizes,
            activation=activation,
            init_layers_orthogonal=init_layers_orthogonal,
        )
    else:
        raise ValueError(f"Unsupported action space: {action_space}")
