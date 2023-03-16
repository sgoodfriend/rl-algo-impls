import numpy as np
import os
import torch

from typing import Optional, Sequence, TypeVar

from rl_algo_impls.dqn.q_net import QNetwork
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnv,
    VecEnvObs,
    single_observation_space,
    single_action_space,
)

DQNPolicySelf = TypeVar("DQNPolicySelf", bound="DQNPolicy")


class DQNPolicy(Policy):
    def __init__(
        self,
        env: VecEnv,
        hidden_sizes: Sequence[int] = [],
        cnn_feature_dim: int = 512,
        cnn_style: str = "nature",
        cnn_layers_init_orthogonal: Optional[bool] = None,
        impala_channels: Sequence[int] = (16, 32, 32),
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.q_net = QNetwork(
            single_observation_space(env),
            single_action_space(env),
            hidden_sizes,
            cnn_feature_dim=cnn_feature_dim,
            cnn_style=cnn_style,
            cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
            impala_channels=impala_channels,
        )

    def act(
        self, obs: VecEnvObs, eps: float = 0, deterministic: bool = True
    ) -> np.ndarray:
        assert eps == 0 if deterministic else eps >= 0
        if not deterministic and np.random.random() < eps:
            return np.array(
                [
                    single_action_space(self.env).sample()
                    for _ in range(self.env.num_envs)
                ]
            )
        else:
            o = self._as_tensor(obs)
            with torch.no_grad():
                return self.q_net(o).argmax(axis=1).cpu().numpy()
