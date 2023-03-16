import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from gym.spaces import Box, Discrete
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from typing import Dict, Optional, Sequence, Type

from rl_algo_impls.shared.module.module import layer_init


class CnnFeatureExtractor(nn.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        in_channels: int,
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__()


class NatureCnn(CnnFeatureExtractor):
    """
    CNN from DQN Nature paper: Mnih, Volodymyr, et al.
    "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.
    """

    def __init__(
        self,
        in_channels: int,
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: Optional[bool] = None,
        **kwargs,
    ) -> None:
        if init_layers_orthogonal is None:
            init_layers_orthogonal = True
        super().__init__(in_channels, activation, init_layers_orthogonal)
        self.cnn = nn.Sequential(
            layer_init(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.cnn(obs)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: bool = False,
    ) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            activation(),
            layer_init(
                nn.Conv2d(channels, channels, 3, padding=1), init_layers_orthogonal
            ),
            activation(),
            layer_init(
                nn.Conv2d(channels, channels, 3, padding=1), init_layers_orthogonal
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)


class ConvSequence(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: bool = False,
    ) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            layer_init(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                init_layers_orthogonal,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResidualBlock(out_channels, activation, init_layers_orthogonal),
            ResidualBlock(out_channels, activation, init_layers_orthogonal),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ImpalaCnn(CnnFeatureExtractor):
    """
    IMPALA-style CNN architecture
    """

    def __init__(
        self,
        in_channels: int,
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: Optional[bool] = None,
        impala_channels: Sequence[int] = (16, 32, 32),
        **kwargs,
    ) -> None:
        if init_layers_orthogonal is None:
            init_layers_orthogonal = False
        super().__init__(in_channels, activation, init_layers_orthogonal)
        sequences = []
        for out_channels in impala_channels:
            sequences.append(
                ConvSequence(
                    in_channels, out_channels, activation, init_layers_orthogonal
                )
            )
            in_channels = out_channels
        sequences.extend(
            [
                activation(),
                nn.Flatten(),
            ]
        )
        self.seq = nn.Sequential(*sequences)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.seq(obs)


CNN_EXTRACTORS_BY_STYLE: Dict[str, Type[CnnFeatureExtractor]] = {
    "nature": NatureCnn,
    "impala": ImpalaCnn,
}


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        obs_space: gym.Space,
        activation: Type[nn.Module],
        init_layers_orthogonal: bool = False,
        cnn_feature_dim: int = 512,
        cnn_style: str = "nature",
        cnn_layers_init_orthogonal: Optional[bool] = None,
        impala_channels: Sequence[int] = (16, 32, 32),
    ) -> None:
        super().__init__()
        if isinstance(obs_space, Box):
            # Conv2D: (channels, height, width)
            if len(obs_space.shape) == 3:
                cnn = CNN_EXTRACTORS_BY_STYLE[cnn_style](
                    obs_space.shape[0],
                    activation,
                    init_layers_orthogonal=cnn_layers_init_orthogonal,
                    impala_channels=impala_channels,
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
