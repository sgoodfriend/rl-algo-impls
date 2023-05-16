from typing import Any, Dict, List

import numpy as np
from luxai_s2.utils import my_turn_to_place_factory

from rl_algo_impls.lux.shared import LuxGameState, pos_to_numpy


def bid_action(bid_std_dev: float, faction: str) -> Dict[str, Any]:
    return {"bid": int(np.random.normal(scale=bid_std_dev)), "faction": faction}
