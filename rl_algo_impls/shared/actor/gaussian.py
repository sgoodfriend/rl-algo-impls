from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal

from rl_algo_impls.shared.actor.actor import Actor, PiForward, pi_forward
from rl_algo_impls.shared.module.utils import mlp


class GaussianDistribution(Normal):
    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        return super().log_prob(a).sum(axis=-1)

    def sample(self) -> torch.Tensor:
        return self.rsample()


class GaussianActorHead(Actor):
    def __init__(
        self,
        act_dim: int,
        in_dim: int,
        hidden_sizes: Tuple[int, ...] = (32,),
        activation: Type[nn.Module] = nn.Tanh,
        init_layers_orthogonal: bool = True,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.act_dim = act_dim
        layer_sizes = (in_dim,) + hidden_sizes + (act_dim,)
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
        return pi_forward(pi, actions)

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return (self.act_dim,)
