from typing import Any, Dict, NamedTuple, Type, TypeVar

import numpy as np

from rl_algo_impls.lux.stats import StatsTracking
from rl_algo_impls.utils.interpolate import InterpolateMethod, interpolate

LuxRewardWeightsSelf = TypeVar("LuxRewardWeightsSelf", bound="LuxRewardWeights")


class LuxRewardWeights(NamedTuple):
    # End-game rewards
    win_loss: float = 0
    score_vs_opponent: float = 0  # scaled between -1 and 1
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
    def interpolate(
        cls: Type[LuxRewardWeightsSelf],
        start: Dict[str, float],
        end: Dict[str, float],
        progress: float,
        method: InterpolateMethod,
    ) -> LuxRewardWeightsSelf:
        return cls(
            *interpolate(np.array(cls(**start)), np.array(cls(**end)), progress, method)
        )


MIN_SCORE = -1000


def from_lux_rewards(
    lux_rewards: Dict[str, float],
    done: bool,
    info: Dict[str, Any],
    stats: StatsTracking,
    reward_weights: LuxRewardWeights,
    verify: bool,
) -> np.ndarray:
    agents = list(lux_rewards)
    if done:
        _win_loss = np.zeros(2, dtype=np.int32)
        for i in range(2):
            own_r = lux_rewards[agents[i]]
            opp_r = lux_rewards[agents[(i + 1) % 2]]
            if own_r > opp_r:
                _win_loss[i] = 1
            elif own_r < opp_r:
                _win_loss[i] = -1
            else:
                _win_loss[i] = 0

        _score_reward = np.zeros(2, dtype=np.float32)
        for idx, agent in enumerate(agents):
            agent_stats = stats.agent_stats[idx]
            action_stats = stats.action_stats[idx]
            info[agent]["stats"] = {
                **info[agent].get("stats", {}),
                **dict(zip(agent_stats.NAMES, agent_stats.stats.tolist())),
            }
            info[agent]["stats"].update(action_stats.stats_dict(prefix="actions_"))
            self_score = lux_rewards[agent]
            opp_score = lux_rewards[agents[(idx + 1) % 2]]
            score_delta = self_score - opp_score
            _score_reward[idx] = score_delta / (
                self_score + opp_score + 1 - 2 * MIN_SCORE
            )
            assert np.sign(_score_reward[idx]) == np.sign(
                _win_loss[idx]
            ), f"score_reward {_score_reward[idx]} must be same sign as winloss {_win_loss[idx]}"
            info[agent]["results"] = {
                "WinLoss": _win_loss[idx],
                "win": int(_win_loss[idx] == 1),
                "loss": int(_win_loss[idx] == -1),
                "score": self_score,
                "score_delta": score_delta,
                "score_reward": _score_reward[idx],
            }
        _done_rewards = np.concatenate(
            [
                np.expand_dims(_win_loss, axis=-1),
                np.expand_dims(_score_reward, axis=-1),
            ],
            axis=-1,
        )
    else:
        _done_rewards = np.zeros((2, 2))
    _stats_delta = stats.update(verify)
    raw_rewards = np.concatenate(
        [
            _done_rewards,
            _stats_delta,
        ],
        axis=-1,
    )
    return np.sum(raw_rewards * np.array(reward_weights), axis=-1)
