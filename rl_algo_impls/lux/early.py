from typing import Any, Dict, List

import numpy as np


def bid_action(bid_std_dev: float, faction: str) -> Dict[str, Any]:
    return {"bid": int(np.random.normal(scale=bid_std_dev)), "faction": faction}
