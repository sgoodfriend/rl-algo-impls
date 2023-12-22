from typing import Optional, Sequence, TypeVar

import numpy as np
import torch

from rl_algo_impls.dqn.q_net import QNetwork
from rl_algo_impls.shared.policy.policy import EnvSpaces, Policy
from rl_algo_impls.wrappers.vector_wrapper import ObsType

DQNPolicySelf = TypeVar("DQNPolicySelf", bound="DQNPolicy")


class DQNPolicy(Policy):
    def __init__(
        self,
        env_spaces: EnvSpaces,
        hidden_sizes: Sequence[int] = [],
        cnn_flatten_dim: int = 512,
        cnn_style: str = "nature",
        cnn_layers_init_orthogonal: Optional[bool] = None,
        impala_channels: Sequence[int] = (16, 32, 32),
        **kwargs,
    ) -> None:
        super().__init__(env_spaces, **kwargs)
        self.q_net = QNetwork(
            env_spaces.single_observation_space,
            env_spaces.single_action_space,
            hidden_sizes,
            cnn_flatten_dim=cnn_flatten_dim,
            cnn_style=cnn_style,
            cnn_layers_init_orthogonal=cnn_layers_init_orthogonal,
            impala_channels=impala_channels,
        )

    def act(
        self,
        obs: ObsType,
        eps: float = 0,
        deterministic: bool = True,
        action_masks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        assert eps == 0 if deterministic else eps >= 0
        assert (
            action_masks is None
        ), f"action_masks not currently supported in {self.__class__.__name__}"
        if not deterministic and np.random.random() < eps:
            return np.array(
                [
                    self.env_spaces.single_action_space.sample()
                    for _ in range(self.env.num_envs)
                ]
            )
        else:
            o = self._as_tensor(obs)
            with torch.no_grad():
                return self.q_net(o).argmax(axis=1).cpu().numpy()
