import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym.spaces import Box, Discrete
from typing import Callable, NamedTuple, Optional, Sequence, Type


class FeatureExtractor(NamedTuple):
    preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]]
    feature_extractor: nn.Module


def feature_extractor(
    obs_space: gym.Space,
    activation: Type[nn.Module],
    features_dim: int,
    init_layers_orthogonal: bool = False,
) -> FeatureExtractor:
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
                    nn.Conv2d(32, 64, kernel_size=4, stride=2), init_layers_orthogonal
                ),
                activation(),
                layer_init(
                    nn.Conv2d(64, 64, kernel_size=3, stride=1), init_layers_orthogonal
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
            return FeatureExtractor(
                preprocess,
                nn.Sequential(
                    cnn,
                    layer_init(
                        nn.Linear(cnn_out.shape[1], features_dim),
                        init_layers_orthogonal,
                    ),
                    activation(),
                ),
            )
        elif len(obs_space.shape) == 1:

            def preprocess(obs: torch.Tensor) -> torch.Tensor:
                if len(obs.shape) == 1:
                    obs = obs.unsqueeze(0)
                return obs.float()

            return FeatureExtractor(
                preprocess,
                nn.Sequential(
                    layer_init(
                        nn.Linear(obs_space.shape[0], features_dim),
                        init_layers_orthogonal,
                    ),
                    activation(),
                ),
            )
        else:
            raise ValueError(f"Unsupported observation space: {obs_space}")
    elif isinstance(obs_space, Discrete):
        return FeatureExtractor(
            lambda x: F.one_hot(x, obs_space.n).float(),
            nn.Sequential(
                layer_init(
                    nn.Linear(obs_space.n, features_dim), init_layers_orthogonal
                ),
                activation(),
            ),
        )
    else:
        raise NotImplementedError


def mlp(
    layer_sizes: Sequence[int],
    activation: Type[nn.Module],
    output_activation: Type[nn.Module] = nn.Identity,
    init_layers_orthogonal: bool = False,
    final_layer_gain: float = np.sqrt(2),
) -> nn.Module:
    layers = []
    for i in range(len(layer_sizes) - 2):
        layers.append(
            layer_init(
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]), init_layers_orthogonal
            )
        )
        layers.append(activation())
    layers.append(
        layer_init(
            nn.Linear(layer_sizes[-2], layer_sizes[-1]),
            init_layers_orthogonal,
            std=final_layer_gain,
        )
    )
    layers.append(output_activation())
    return nn.Sequential(*layers)


def layer_init(
    layer: nn.Module, init_layers_orthogonal: bool, std: float = np.sqrt(2)
) -> nn.Module:
    if not init_layers_orthogonal:
        return layer
    nn.init.orthogonal_(layer.weight, std)  # type: ignore
    nn.init.constant_(layer.bias, 0.0)  # type: ignore
    return layer
