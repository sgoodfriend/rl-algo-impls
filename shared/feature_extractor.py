import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym.spaces import Box, Discrete
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from typing import Type

from shared.module import layer_init


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        obs_space: gym.Space,
        activation: Type[nn.Module],
        init_layers_orthogonal: bool = False,
        cnn_feature_dim: int = 512,
    ) -> None:
        super().__init__()
        if isinstance(obs_space, Box):
            # Conv2D: (channels, height, width)
            if len(obs_space.shape) == 3:
                # CNN from DQN Nature paper: Mnih, Volodymyr, et al.
                # "Human-level control through deep reinforcement learning."
                # Nature 518.7540 (2015): 529-533.
                cnn = nn.Sequential(
                    layer_init(
                        nn.Conv2d(obs_space.shape[0], 32, kernel_size=8, stride=4),
                        init_layers_orthogonal,
                    ),
                    activation(),
                    layer_init(
                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                        init_layers_orthogonal,
                    ),
                    activation(),
                    layer_init(
                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                        init_layers_orthogonal,
                    ),
                    activation(),
                    nn.Flatten(),
                )

                def preprocess(obs: torch.Tensor) -> torch.Tensor:
                    if len(obs.shape) == 3:
                        obs = obs.unsqueeze(0)
                    return obs.float() / 255.0

                with torch.no_grad():
                    cnn_out = cnn(preprocess(torch.as_tensor(obs_space.sample())))
                self.preprocess = preprocess
                self.feature_extractor = nn.Sequential(
                    cnn,
                    layer_init(
                        nn.Linear(cnn_out.shape[1], cnn_feature_dim),
                        init_layers_orthogonal,
                    ),
                    activation(),
                )
                self.out_dim = cnn_feature_dim
            elif len(obs_space.shape) == 1:

                def preprocess(obs: torch.Tensor) -> torch.Tensor:
                    if len(obs.shape) == 1:
                        obs = obs.unsqueeze(0)
                    return obs.float()

                self.preprocess = preprocess
                self.feature_extractor = nn.Flatten()
                self.out_dim = get_flattened_obs_dim(obs_space)
            else:
                raise ValueError(f"Unsupported observation space: {obs_space}")
        elif isinstance(obs_space, Discrete):
            self.preprocess = lambda x: F.one_hot(x, obs_space.n).float()
            self.feature_extractor = nn.Flatten()
            self.out_dim = obs_space.n
        else:
            raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.preprocess:
            obs = self.preprocess(obs)
        return self.feature_extractor(obs)
