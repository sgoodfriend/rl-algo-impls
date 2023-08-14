import dataclasses
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

from rl_algo_impls.lux.actions import FACTORY_ACTION_ENCODED_SIZE, UNIT_ACTION_SIZES


@dataclass
class ActionMatchingStats:
    action_match_cnt = 0
    action_mismatch_cnt = 0

    def stats_dict(self) -> Dict[str, float]:
        return {
            f"action_match_fraction": self.action_match_cnt
            / (self.action_match_cnt + self.action_mismatch_cnt)
            if self.action_match_cnt
            else 0
        }


@dataclass
class FactoryActionStats:
    ACTION_NAMES = ("nothing", "built_light", "built_heavy", "grow_lichen")
    action_type: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(4, dtype=np.int32)
    )

    def stats_dict(self, prefix: str) -> Dict[str, int]:
        return {
            f"{prefix}{name}": cnt
            for name, cnt in zip(self.ACTION_NAMES[1:], self.action_type[1:])
        }


@dataclass
class UnitActionStats:
    ACTION_NAMES = ("move", "transfer", "pickup", "dig", "self_destruct", "recharge")
    action_type: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(6, dtype=np.int32)
    )

    def stats_dict(self, prefix: str) -> Dict[str, int]:
        return {
            f"{prefix}{name}": cnt
            for name, cnt in zip(self.ACTION_NAMES, self.action_type)
        }


class ReplayActionStats:
    def __init__(self) -> None:
        self.replay_factory = FactoryActionStats()
        self.replay_unit = UnitActionStats()
        self.policy_factory = FactoryActionStats()
        self.policy_unit = UnitActionStats()
        self.matching = ActionMatchingStats()

    def stats_dict(self) -> Dict[str, Union[float, int]]:
        return {
            **self.replay_factory.stats_dict("replay_"),
            **self.replay_unit.stats_dict("replay_"),
            **self.policy_factory.stats_dict("policy_"),
            **self.policy_unit.stats_dict("policy_"),
            **self.matching.stats_dict(),
        }

    def update_action_stats(
        self,
        replay_action: Dict[str, np.ndarray],
        policy_action: Optional[Dict[str, np.ndarray]],
        action_mask: Dict[str, np.ndarray],
    ) -> None:
        PER_POSITION_KEY = "per_position"
        r_action = replay_action[PER_POSITION_KEY]
        p_action = policy_action[PER_POSITION_KEY] if policy_action else None
        mask = action_mask[PER_POSITION_KEY]

        rf_action_type, rf_selected, rf_valid, rf_count = factory_action_arrays(
            r_action, mask
        )
        self.replay_factory.action_type += rf_count
        r_unit_action_type, r_selected, r_valid, r_count = unit_action_arrays(
            r_action, mask
        )
        self.replay_unit.action_type += r_count

        if p_action is not None:
            pf_action_type, _, _, pf_count = factory_action_arrays(p_action, mask)
            self.policy_factory.action_type += pf_count
            p_unit_action_type, _, _, p_count = unit_action_arrays(p_action, mask)
            self.policy_unit.action_type += p_count

            rp_factory_match = np.sum(rf_selected & (rf_action_type == pf_action_type))
            rp_unit_match = np.sum(
                r_selected & (r_unit_action_type == p_unit_action_type)
            )
            self.matching.action_match_cnt += rp_factory_match + rp_unit_match
            self.matching.action_mismatch_cnt += (len(r_valid) - rp_unit_match) + (
                len(rf_valid) - rp_factory_match
            )


def factory_action_arrays(
    action: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    factory_action = action[:, 0]
    selected = mask[np.arange(len(factory_action)), factory_action]
    valid = factory_action[selected]
    count = np.bincount(valid, minlength=FACTORY_ACTION_ENCODED_SIZE)
    return factory_action, selected, valid, count


def unit_action_arrays(
    action: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unit_action_type = action[:, 1]
    selected = mask[
        np.arange(len(unit_action_type)), unit_action_type + FACTORY_ACTION_ENCODED_SIZE
    ]
    valid = unit_action_type[selected]
    count = np.bincount(valid, minlength=UNIT_ACTION_SIZES[0])
    return unit_action_type, selected, valid, count
