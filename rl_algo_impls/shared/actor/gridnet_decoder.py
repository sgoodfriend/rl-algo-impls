from typing import Dict, Optional, Sequence, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.distributions import Distribution, constraints

from rl_algo_impls.shared.actor import Actor, PiForward
from rl_algo_impls.shared.actor.categorical import MaskedCategorical
from rl_algo_impls.shared.module.module import mlp


class GridnetDecoder(Actor):
    pass
