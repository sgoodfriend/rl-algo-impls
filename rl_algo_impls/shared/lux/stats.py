import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from luxai_s2.env import LuxAI_S2
from luxai_s2.unit import UnitType


class AgentRunningStats:
    stats: np.ndarray
    NAMES = (
        # Change in value stats
        "ice_generation",
        "ore_generation",
        "water_generation",
        "metal_generation",
        "power_generation",
        "lichen",
        "built_light",
        "built_heavy",
        "lost_factory",
        # Current value stats
        "factories_alive",
        "heavies_alive",
        "lights_alive",
    )

    def __init__(self) -> None:
        self.stats = np.zeros(len(self.NAMES), dtype=np.int32)

    def update(self, env: LuxAI_S2, agent: str) -> np.ndarray:
        generation = env.state.stats[agent]["generation"]
        strain_ids = env.state.teams[agent].factory_strains
        agent_lichen_mask = np.isin(env.state.board.lichen_strains, strain_ids)
        lichen = env.state.board.lichen[agent_lichen_mask].sum()

        new_delta_stats = np.array(
            [
                sum(generation["ice"].values()),
                sum(generation["ore"].values()),
                generation["water"],
                generation["metal"],
                sum(generation["power"].values()),
                lichen,
                generation["built"]["LIGHT"],
                generation["built"]["HEAVY"],
                env.state.stats[agent]["destroyed"]["FACTORY"],
            ]
        )
        delta = new_delta_stats - self.stats[: len(new_delta_stats)]

        agent_units = env.state.units[agent]
        new_current_stats = np.array(
            [
                len(env.state.factories[agent]),
                len([u for u in agent_units.values() if u.unit_type == UnitType.HEAVY]),
                len([u for u in agent_units.values() if u.unit_type == UnitType.LIGHT]),
            ]
        )

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
        per_agent_updates = np.stack(
            [
                self.agent_stats[idx].update(self.env, agent)
                for idx, agent in enumerate(self.agents)
            ]
        )
        lichen_idx = AgentRunningStats.NAMES.index("lichen")
        delta_vs_opponent = np.expand_dims(
            np.array(
                [
                    per_agent_updates[p_idx, lichen_idx]
                    - per_agent_updates[o_idx, lichen_idx]
                    for p_idx, o_idx in zip(
                        range(len(self.agents)),
                        reversed(range(len(self.agents))),
                    )
                ]
            ),
            axis=-1,
        )
        return np.concatenate([per_agent_updates, delta_vs_opponent], axis=-1)

    def reset(self, env: LuxAI_S2) -> None:
        self.env = env
        self.agents = env.agents
        self.agent_stats = (AgentRunningStats(), AgentRunningStats())
        self.action_stats = (ActionStats(), ActionStats())
        self.update()
