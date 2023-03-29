from typing import Dict, Optional, Sequence, Type

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

from rl_algo_impls.shared.encoder.cnn import CnnFeatureExtractor
from rl_algo_impls.shared.encoder.impala_cnn import ImpalaCnn
from rl_algo_impls.shared.encoder.microrts_cnn import MicrortsCnn
from rl_algo_impls.shared.encoder.nature_cnn import NatureCnn
from rl_algo_impls.shared.module.module import layer_init

CNN_EXTRACTORS_BY_STYLE: Dict[str, Type[CnnFeatureExtractor]] = {
    "nature": NatureCnn,
    "impala": ImpalaCnn,
    "microrts": MicrortsCnn,
}


class Encoder(nn.Module):
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

                range_size = np.max(obs_space.high) - np.min(obs_space.low)

                def preprocess(obs: torch.Tensor) -> torch.Tensor:
                    if len(obs.shape) == 3:
                        obs = obs.unsqueeze(0)
                    return obs.float() / range_size

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
