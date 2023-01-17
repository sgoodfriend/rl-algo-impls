import numpy as np
import torch

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs
from typing import Sequence, TypeVar

from dqn.q_net import QNetwork
from shared.policy import Policy

DQNPolicySelf = TypeVar("DQNPolicySelf", bound="DQNPolicy")


class DQNPolicy(Policy):
    def __init__(
        self,
        env: VecEnv,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__(env)
        self.q_net = QNetwork(env.observation_space, env.action_space, hidden_sizes)

    def act(self, obs: VecEnvObs, eps: float = 0) -> np.ndarray:
        if self.training and np.random.random() < eps:
            return np.array(
                [self.env.action_space.sample() for _ in range(self.env.num_envs)]
            )
        else:
            with torch.no_grad():
                obs_th = torch.as_tensor(np.array(obs))
                if self.device:
                    obs_th = obs_th.to(self.device)
                return self.q_net(obs_th).argmax(axis=1).cpu().numpy()

    def train(self: DQNPolicySelf, mode: bool = True) -> DQNPolicySelf:
        super().train(mode)
        self.q_net.train(mode)
        return self

    def save(self, path: str) -> None:
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.q_net.load_state_dict(torch.load(path))
