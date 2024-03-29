from typing import Optional, Sequence, Type

import numpy as np
import torch.nn as nn

from rl_algo_impls.shared.module.normalization import normalization1d


def mlp(
    layer_sizes: Sequence[int],
    activation: Type[nn.Module],
    output_activation: Optional[nn.Module] = None,
    init_layers_orthogonal: bool = False,
    final_layer_gain: float = np.sqrt(2),
    hidden_layer_gain: float = np.sqrt(2),
    normalization: Optional[str] = None,
    final_normalization: Optional[str] = None,
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
        if normalization:
            layers.append(normalization1d(normalization, layer_sizes[i + 1]))
    layers.append(
        layer_init(
            nn.Linear(layer_sizes[-2], layer_sizes[-1]),
            init_layers_orthogonal,
            std=final_layer_gain,
        )
    )
    if output_activation is not None:
        layers.append(output_activation)
    if final_normalization:
        layers.append(normalization1d(final_normalization, layer_sizes[-1]))
    return nn.Sequential(*layers)


def layer_init(
    layer: nn.Module,
    init_layers_orthogonal: bool,
    std: float = np.sqrt(2),
) -> nn.Module:
    if not init_layers_orthogonal:
        return layer
    nn.init.orthogonal_(layer.weight, std)  # type: ignore
    nn.init.constant_(layer.bias, 0.0)  # type: ignore
    return layer
