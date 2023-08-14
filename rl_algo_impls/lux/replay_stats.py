import dataclasses
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Union

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

        for idx, (ra, mask) in enumerate(
            zip(replay_action[PER_POSITION_KEY], action_mask[PER_POSITION_KEY])
        ):
            record_action(ra, mask, self.replay_factory, self.replay_unit)
            if policy_action is not None:
                pa = policy_action[PER_POSITION_KEY][idx]
                record_action(pa, mask, self.policy_factory, self.policy_unit)
                if ra[0] == pa[0]:
                    if ra[0] != 0:
                        self.matching.action_match_cnt += 1
                else:
                    self.matching.action_mismatch_cnt += 1
                unit_action_mask = mask[
                    FACTORY_ACTION_ENCODED_SIZE : FACTORY_ACTION_ENCODED_SIZE
                    + UNIT_ACTION_SIZES[0]
                ]
                if unit_action_mask.any():
                    if ra[1] == pa[1]:
                        self.matching.action_match_cnt += 1
                    else:
                        self.matching.action_mismatch_cnt += 1


def record_action(
    action: np.ndarray,
    action_mask: np.ndarray,
    factory_stats: FactoryActionStats,
    unit_stats: UnitActionStats,
):
    if action[0]:
        factory_stats.action_type[action[0]] += 1
    unit_action_mask = action_mask[
        FACTORY_ACTION_ENCODED_SIZE : FACTORY_ACTION_ENCODED_SIZE + UNIT_ACTION_SIZES[0]
    ]
    if unit_action_mask.any():
        if unit_action_mask[action[1]]:
            unit_stats.action_type[action[1]] += 1
        else:
            logging.warning(
                f"Illegal action {action[1]}. Legal actions: {unit_action_mask}"
            )
