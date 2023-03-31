from typing import Dict, Optional, Sequence, Type

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

from rl_algo_impls.shared.encoder.cnn import CnnEncoder
from rl_algo_impls.shared.encoder.gridnet_encoder import GridnetEncoder
from rl_algo_impls.shared.encoder.impala_cnn import ImpalaCnn
from rl_algo_impls.shared.encoder.microrts_cnn import MicrortsCnn
from rl_algo_impls.shared.encoder.nature_cnn import NatureCnn
from rl_algo_impls.shared.module.utils import layer_init

CNN_EXTRACTORS_BY_STYLE: Dict[str, Type[CnnEncoder]] = {
    "nature": NatureCnn,
    "impala": ImpalaCnn,
    "microrts": MicrortsCnn,
    "gridnet_encoder": GridnetEncoder,
}


class Encoder(nn.Module):
    def __init__(
        self,
        obs_space: gym.Space,
        activation: Type[nn.Module],
        init_layers_orthogonal: bool = False,
        cnn_flatten_dim: int = 512,
        cnn_style: str = "nature",
        cnn_layers_init_orthogonal: Optional[bool] = None,
        impala_channels: Sequence[int] = (16, 32, 32),
    ) -> None:
        super().__init__()
        if isinstance(obs_space, Box):
            # Conv2D: (channels, height, width)
            if len(obs_space.shape) == 3:  # type: ignore
                self.preprocess = None
                cnn = CNN_EXTRACTORS_BY_STYLE[cnn_style](
                    obs_space,
                    activation=activation,
                    cnn_init_layers_orthogonal=cnn_layers_init_orthogonal,
                    linear_init_layers_orthogonal=init_layers_orthogonal,
                    cnn_flatten_dim=cnn_flatten_dim,
                    impala_channels=impala_channels,
                )
                self.feature_extractor = cnn
                self.out_dim = cnn.out_dim
            elif len(obs_space.shape) == 1:  # type: ignore

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
            self.out_dim = obs_space.n  # type: ignore
        else:
            raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.preprocess:
            obs = self.preprocess(obs)
        return self.feature_extractor(obs)
