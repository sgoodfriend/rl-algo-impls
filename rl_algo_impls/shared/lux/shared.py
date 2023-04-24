from typing import Union

from luxai_s2.config import EnvConfig
from luxai_s2.state import State

from rl_algo_impls.lux.kit.config import EnvConfig as KitEnvConfig
from rl_algo_impls.lux.kit.kit import GameState

LuxEnvConfig = Union[EnvConfig, KitEnvConfig]
LuxGameState = Union[State, GameState]
