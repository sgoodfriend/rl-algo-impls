from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from gym.spaces import Space

from rl_algo_impls.shared.actor import actor_head
from rl_algo_impls.shared.encoder import Encoder
from rl_algo_impls.shared.policy.actor_critic_network.network import (
    ACNForward,
    ActorCriticNetwork,
    default_hidden_sizes,
)
from rl_algo_impls.shared.policy.critic import CriticHead
from rl_algo_impls.shared.policy.policy import ACTIVATION


class UNetActorCriticNetwork(ActorCriticNetwork):
    ...
