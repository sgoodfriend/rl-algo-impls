from typing import Sequence, Type

import numpy as np
import torch.nn as nn


def mlp(
    layer_sizes: Sequence[int],
    activation: Type[nn.Module],
    output_activation: Type[nn.Module] = nn.Identity,
    init_layers_orthogonal: bool = False,
    final_layer_gain: float = np.sqrt(2),
    hidden_layer_gain: float = np.sqrt(2),
) -> nn.Module:
    layers = []
    for i in range(len(layer_sizes) - 2):
        layers.append(
            layer_init(
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                init_layers_orthogonal,
                std=hidden_layer_gain,
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
