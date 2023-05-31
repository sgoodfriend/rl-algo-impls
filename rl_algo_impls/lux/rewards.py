from typing import Dict, NamedTuple, Type, TypeVar

import numpy as np

from rl_algo_impls.shared.schedule import lerp

LuxRewardWeightsSelf = TypeVar("LuxRewardWeightsSelf", bound="LuxRewardWeights")


class LuxRewardWeights(NamedTuple):
    # End-game rewards
    win_loss: float = 0
    score_vs_opponent: float = 0  # clipped between +/- 1
    # Change in value stats
    ice_generation: float = 0
    ore_generation: float = 0
    water_generation: float = 0
    metal_generation: float = 0
    power_generation: float = 0
    lichen_delta: float = 0
    built_light: float = 0
    built_heavy: float = 0
    lost_factory: float = 0
    # Accumulation stats
    ice_rubble_cleared: float = 0
    ore_rubble_cleared: float = 0
    # Current value stats
    factories_alive: float = 0
    heavies_alive: float = 0
    lights_alive: float = 0
    # Change in value stats vs opponent
    lichen_delta_vs_opponent: float = 0

    @classmethod
    def sparse(cls: Type[LuxRewardWeightsSelf]) -> LuxRewardWeightsSelf:
        return cls(win_loss=1)

    @classmethod
    def default_start(cls: Type[LuxRewardWeightsSelf]) -> LuxRewardWeightsSelf:
        return cls(
            ice_generation=0.01,  # 2 for a day of water for factory, 0.2 for a heavy dig action
            ore_generation=2e-3,  # 1 for building a heavy robot, 0.04 for a heavy dig action
            water_generation=0.04,  # 2 for a day of water for factory
            metal_generation=0.01,  # 1 for building a heavy robot, 0.04 for a heavy dig action
            power_generation=0.0004,  # factory 1/day, heavy 0.12/day, light 0.012/day, lichen 0.02/day
            lost_factory=-1,
            ice_rubble_cleared=0.008,  # 80% of ice_generation
            ore_rubble_cleared=1.8e-3,  # 90% of ore_generation
        )

    @classmethod
    def lerp(
        cls: Type[LuxRewardWeightsSelf],
        start: Dict[str, float],
        end: Dict[str, float],
        progress: float,
    ) -> LuxRewardWeightsSelf:
        return cls(*lerp(np.array(cls(**start)), np.array(cls(**end)), progress))
