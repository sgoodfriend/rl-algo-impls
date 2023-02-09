import numpy as np
import torch
import torch.nn as nn

from gym.spaces import Box
from pathlib import Path
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs
from typing import NamedTuple, Optional, Sequence, TypeVar

from shared.module import FeatureExtractor
from shared.policy.actor import (
    PiForward,
    Actor,
    StateDependentNoiseActorHead,
    actor_head,
)
from shared.policy.critic import CriticHead
from shared.policy.on_policy import Step, clamp_actions, default_hidden_sizes
from shared.policy.policy import ACTIVATION, Policy

PI_FILE_NAME = "pi.pt"
V_FILE_NAME = "v.pt"


class VPGActor(Actor):
    def __init__(self, feature_extractor: FeatureExtractor, head: Actor) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, obs: torch.Tensor, a: Optional[torch.Tensor] = None) -> PiForward:
        fe = self.feature_extractor(obs)
        return self.head(fe, a)


class VPGActorCritic(Policy):
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
        obs_space = env.observation_space
        self.action_space = env.action_space
        self.use_sde = use_sde
        self.squash_output = squash_output

        hidden_sizes = (
            hidden_sizes
            if hidden_sizes is not None
            else default_hidden_sizes(obs_space)
        )

        pi_feature_extractor = FeatureExtractor(
            obs_space, activation, init_layers_orthogonal=init_layers_orthogonal
        )
        pi_head = actor_head(
            self.action_space,
            (pi_feature_extractor.out_dim,) + tuple(hidden_sizes),
            init_layers_orthogonal,
            activation,
            log_std_init=log_std_init,
            use_sde=use_sde,
            full_std=full_std,
            squash_output=squash_output,
        )
        self.pi = VPGActor(pi_feature_extractor, pi_head)

        v_feature_extractor = FeatureExtractor(
            obs_space, activation, init_layers_orthogonal=init_layers_orthogonal
        )
        v_head = CriticHead(
            (v_feature_extractor.out_dim,) + tuple(hidden_sizes),
            activation=activation,
            init_layers_orthogonal=init_layers_orthogonal,
        )
        self.v = nn.Sequential(v_feature_extractor, v_head)

    def _as_tensor(self, obs: VecEnvObs) -> torch.Tensor:
        assert isinstance(obs, np.ndarray)
        o = torch.as_tensor(obs)
        if self.device is not None:
            o = o.to(self.device)
        return o

    def step(self, obs: VecEnvObs) -> Step:
        o = self._as_tensor(obs)
        with torch.no_grad():
            pi, _, _ = self.pi(o)
            a = pi.sample()
            logp_a = pi.log_prob(a)

            v = self.v(o)

        a_np = a.cpu().numpy()
        clamped_a_np = clamp_actions(a_np, self.action_space, self.squash_output)
        return Step(a_np, v.cpu().numpy(), logp_a.cpu().numpy(), clamped_a_np)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
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
        if isinstance(self.pi.head, StateDependentNoiseActorHead):
            self.pi.head.sample_weights(
                batch_size=batch_size if batch_size else self.env.num_envs
            )
