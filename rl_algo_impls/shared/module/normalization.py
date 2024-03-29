from abc import ABC
from enum import Enum

import torch
import torch.nn as nn

from rl_algo_impls.shared.module.channel_layer_norm import ChannelLayerNorm2d


class NormalizationMethod(Enum):
    BATCH = 1
    LAYER = 2
    BATCH_RENORM = 3


def normalization1d(method_name: str, num_features: int) -> nn.Module:
    norm_method = NormalizationMethod[method_name.upper()]
    if norm_method == NormalizationMethod.BATCH:
        norm_class = nn.BatchNorm1d
    elif norm_method == NormalizationMethod.LAYER:
        norm_class = nn.LayerNorm
    elif norm_method == NormalizationMethod.BATCH_RENORM:
        from batchrenorm import BatchRenorm1d

        norm_class = BatchRenorm1d
    else:
        raise ValueError(f"Unknown normalization method {method_name}")
    return norm_class(num_features)


def normalization2d(method_name: str, num_features: int) -> nn.Module:
    norm_method = NormalizationMethod[method_name.upper()]
    if norm_method == NormalizationMethod.BATCH:
        norm_class = nn.BatchNorm2d
    elif norm_method == NormalizationMethod.LAYER:
        norm_class = ChannelLayerNorm2d
    elif norm_method == NormalizationMethod.BATCH_RENORM:
        from batchrenorm import BatchRenorm2d

        norm_class = BatchRenorm2d
    else:
        raise ValueError(f"Unknown normalization method {method_name}")
    return norm_class(num_features)
