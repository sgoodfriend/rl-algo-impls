from abc import ABC
from enum import Enum

import torch
import torch.nn as nn
from batchrenorm import BatchRenorm1d, BatchRenorm2d

from rl_algo_impls.shared.module.channel_layer_norm import ChannelLayerNorm2d


class NormalizationMethod(Enum):
    BATCH = 1
    LAYER = 2
    BATCH_RENORM = 3


def normalization1d(method_name: str, num_features: int) -> nn.Module:
    return {
        NormalizationMethod.BATCH: nn.BatchNorm1d,
        NormalizationMethod.LAYER: nn.LayerNorm,
        NormalizationMethod.BATCH_RENORM: BatchRenorm1d,
    }[NormalizationMethod[method_name.upper()]](num_features)


def normalization2d(method_name: str, num_features: int) -> nn.Module:
    return {
        NormalizationMethod.BATCH: nn.BatchNorm2d,
        NormalizationMethod.LAYER: ChannelLayerNorm2d,
        NormalizationMethod.BATCH_RENORM: BatchRenorm2d,
    }[NormalizationMethod[method_name.upper()]](num_features)
