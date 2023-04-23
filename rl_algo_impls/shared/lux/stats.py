import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from luxai_s2.env import LuxAI_S2


class AgentRunningStats:
    stats: np.ndarray
    NAMES = (
        # Change in value stats
        "ice_generation",
        "ore_generation",
        "water_generation",
        "metal_generation",
        "lichen_generation",
        "built_light",
        "built_heavy",
        "lost_factory",
        # Current value stats
        "factories_alive",
    )

    def __init__(self) -> None:
        self.stats = np.zeros(len(self.NAMES), dtype=np.int32)

    def update(self, env: LuxAI_S2, agent: str) -> np.ndarray:
        generation = env.state.stats[agent]["generation"]

        new_delta_stats = np.array(
            [
                sum(generation["ice"].values()),
                sum(generation["ore"].values()),
                generation["water"],
                generation["metal"],
                generation["lichen"],
                generation["built"]["LIGHT"],
                generation["built"]["HEAVY"],
                env.state.stats[agent]["destroyed"]["FACTORY"],
            ]
        )
        delta = new_delta_stats - self.stats[: len(new_delta_stats)]

        new_current_stats = np.array([len(env.state.factories[agent])])

        self.stats = np.concatenate((new_delta_stats, new_current_stats))
        return np.concatenate((delta, new_current_stats))


@dataclass
class ActionStats:
    ACTION_NAMES = ("move", "transfer", "pickup", "dig", "self_destruct", "recharge")
    action_type: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(6, dtype=np.int32)
    )
    no_valid_action = 0
    repeat_action = 0

    def stats_dict(self, prefix: str) -> Dict[str, int]:
        _dict = {
            f"{prefix}{name}": cnt
            for name, cnt in zip(self.ACTION_NAMES, self.action_type.tolist())
        }
        _dict[f"{prefix}no_valid"] = self.no_valid_action
        _dict[f"{prefix}repeat"] = self.repeat_action
        return _dict


class StatsTracking:
    env: LuxAI_S2
    agents: List[str]
    agent_stats: Tuple[AgentRunningStats, AgentRunningStats]
    action_stats: Tuple[ActionStats, ActionStats]

    def update(self) -> np.ndarray:
        return np.stack(
            [
                self.agent_stats[idx].update(self.env, agent)
                for idx, agent in enumerate(self.agents)
            ]
        )

    def reset(self, env: LuxAI_S2) -> None:
        self.env = env
        self.agents = env.agents
        self.agent_stats = (AgentRunningStats(), AgentRunningStats())
        self.action_stats = (ActionStats(), ActionStats())
        self.update()
