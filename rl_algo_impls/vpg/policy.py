from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from rl_algo_impls.shared.actor import Actor, PiForward, actor_head
from rl_algo_impls.shared.encoder import Encoder
from rl_algo_impls.shared.policy.actor_critic import OnPolicy, Step, clamp_actions
from rl_algo_impls.shared.policy.actor_critic_network import default_hidden_sizes
from rl_algo_impls.shared.policy.critic import CriticHead
from rl_algo_impls.shared.policy.policy import ACTIVATION
from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnv,
    VecEnvObs,
    single_action_space,
    single_observation_space,
)

PI_FILE_NAME = "pi.pt"
V_FILE_NAME = "v.pt"


class VPGActor(Actor):
    def __init__(self, feature_extractor: Encoder, head: Actor) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, obs: torch.Tensor, a: Optional[torch.Tensor] = None) -> PiForward:
        fe = self.feature_extractor(obs)
        return self.head(fe, a)

    def sample_weights(self, batch_size: int = 1) -> None:
        self.head.sample_weights(batch_size=batch_size)

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return self.head.action_shape


class VPGActorCritic(OnPolicy):
    def __init__(
        self,
        env: VecEnv,
        hidden_sizes: Optional[Sequence[int]] = None,
        init_layers_orthogonal: bool = True,
        activation_fn: str = "tanh",
        log_std_init: float = -0.5,
        use_sde: bool = False,
        full_std: bool = True,
        squash_output: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        activation = ACTIVATION[activation_fn]
        obs_space = single_observation_space(env)
        self.action_space = single_action_space(env)
        self.use_sde = use_sde
        self.squash_output = squash_output

        hidden_sizes = (
            hidden_sizes
            if hidden_sizes is not None
            else default_hidden_sizes(obs_space)
        )

        pi_feature_extractor = Encoder(
            obs_space, activation, init_layers_orthogonal=init_layers_orthogonal
        )
        pi_head = actor_head(
            self.action_space,
            pi_feature_extractor.out_dim,
            tuple(hidden_sizes),
            init_layers_orthogonal,
            activation,
            log_std_init=log_std_init,
            use_sde=use_sde,
            full_std=full_std,
            squash_output=squash_output,
        )
        self.pi = VPGActor(pi_feature_extractor, pi_head)

        v_feature_extractor = Encoder(
            obs_space, activation, init_layers_orthogonal=init_layers_orthogonal
        )
        v_head = CriticHead(
            v_feature_extractor.out_dim,
            tuple(hidden_sizes),
            activation=activation,
            init_layers_orthogonal=init_layers_orthogonal,
        )
        self.v = nn.Sequential(v_feature_extractor, v_head)

    def value(self, obs: VecEnvObs) -> np.ndarray:
        o = self._as_tensor(obs)
        with torch.no_grad():
            v = self.v(o)
        return v.cpu().numpy()

    def step(self, obs: VecEnvObs, action_masks: Optional[np.ndarray] = None) -> Step:
        assert (
            action_masks is None
        ), f"action_masks not currently supported in {self.__class__.__name__}"
        o = self._as_tensor(obs)
        with torch.no_grad():
            pi, _, _ = self.pi(o)
            a = pi.sample()
            logp_a = pi.log_prob(a)

            v = self.v(o)

        a_np = a.cpu().numpy()
        clamped_a_np = clamp_actions(a_np, self.action_space, self.squash_output)
        return Step(a_np, v.cpu().numpy(), logp_a.cpu().numpy(), clamped_a_np)

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        action_masks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        assert (
            action_masks is None
        ), f"action_masks not currently supported in {self.__class__.__name__}"
        if not deterministic:
            return self.step(obs).clamped_a
        else:
            o = self._as_tensor(obs)
            with torch.no_grad():
                pi, _, _ = self.pi(o)
                a = pi.mode
            return clamp_actions(a.cpu().numpy(), self.action_space, self.squash_output)

    def load(self, path: str) -> None:
        super().load(path)
        self.reset_noise()

    def reset_noise(self, batch_size: Optional[int] = None) -> None:
        self.pi.sample_weights(
            batch_size=batch_size if batch_size else self.env.num_envs
        )

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return self.pi.action_shape
