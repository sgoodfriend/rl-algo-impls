import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from luxai_s2.env import LuxAI_S2
from luxai_s2.unit import UnitType
from luxai_s2.utils.utils import is_day

from rl_algo_impls.lux.shared import LuxGameState, idx_to_pos, pos_to_idx


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
        # Accumalation stats
        "ice_rubble_cleared",
        "ore_rubble_cleared",
        "built_light_by_time_remaining",
        "built_heavy_by_time_remaining",
        # Current value stats
        "factories_alive",
        "heavies_alive",
        "lights_alive",
    )

    def __init__(self, env: LuxAI_S2) -> None:
        self.stats = np.zeros(len(self.NAMES), dtype=np.int32)
        ice_positions = np.argwhere(env.state.board.ice)
        self.rubble_by_ice_pos = rubble_at_positions(env.state, ice_positions)
        ore_positions = np.argwhere(env.state.board.ore)
        self.rubble_by_ore_pos = rubble_at_positions(env.state, ore_positions)

    def update(self, env: LuxAI_S2, agent: str, verify: bool) -> np.ndarray:
        generation = env.state.stats[agent]["generation"]
        strain_ids = env.state.teams[agent].factory_strains
        agent_lichen_mask = np.isin(env.state.board.lichen_strains, strain_ids)
        lichen = env.state.board.lichen[agent_lichen_mask].sum()
        game_remaining = (
            env.state.env_cfg.max_episode_length - env.state.real_env_steps
        ) / (env.state.env_cfg.max_episode_length)

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

        accumulation_stats = np.array(
            [
                update_rubble_cleared_off_positions(
                    env.state, agent, self.rubble_by_ice_pos
                ),
                update_rubble_cleared_off_positions(
                    env.state, agent, self.rubble_by_ore_pos
                ),
                delta[self.NAMES.index("built_light")] * game_remaining,
                delta[self.NAMES.index("built_heavy")] * game_remaining,
            ]
        )
        new_accumulation_stats = (
            self.stats[len(delta) : len(delta) + len(accumulation_stats)]
            + accumulation_stats
        )

        agent_units = env.state.units[agent]
        num_factories = len(env.state.factories[agent])
        num_heavies = len(
            [u for u in agent_units.values() if u.unit_type == UnitType.HEAVY]
        )
        num_lights = len(
            [u for u in agent_units.values() if u.unit_type == UnitType.LIGHT]
        )
        new_current_stats = np.array(
            [
                num_factories,
                num_heavies,
                num_lights,
            ]
        )

        if verify and env.state.real_env_steps > 0:
            cfg = env.state.env_cfg
            min_expected_power_gen = (
                num_factories * cfg.FACTORY_CHARGE
                + cfg.POWER_PER_CONNECTED_LICHEN_TILE
                * sum(
                    len(f.connected_lichen_positions)
                    for f in env.state.factories[agent].values()
                )
            )
            ts = env.state.real_env_steps - 1  # Power gen is before step increment
            max_expected_power_gen = min_expected_power_gen + (
                (
                    num_heavies * cfg.ROBOTS["HEAVY"].CHARGE
                    + num_lights * cfg.ROBOTS["LIGHT"].CHARGE
                )
                if is_day(cfg, ts)
                else 0
            )
            assert (
                min_expected_power_gen <= delta[4] <= max_expected_power_gen
            ), f"Power gen {delta[4]} not between {min_expected_power_gen} and {max_expected_power_gen}"

        self.stats = np.concatenate(
            (new_delta_stats, new_accumulation_stats, new_current_stats)
        )
        return np.concatenate((delta, accumulation_stats, new_current_stats))

    def __getattr__(self, name):
        if name in self.NAMES:
            return self.stats[self.NAMES.index(name)]
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")


def update_rubble_cleared_off_positions(
    state: LuxGameState, agent: str, rubble_by_pos: Dict[int, int]
) -> int:
    rubble_cleared = 0
    own_unit_positions = {
        pos_to_idx(u.pos, state.env_cfg.map_size) for u in state.units[agent].values()
    }
    for pos_idx, r in rubble_by_pos.items():
        pos = idx_to_pos(pos_idx, state.env_cfg.map_size)
        new_r = state.board.rubble[pos[0], pos[1]]
        if new_r < r and pos_idx in own_unit_positions:
            rubble_cleared += r - new_r
        rubble_by_pos[pos_idx] = new_r
    return rubble_cleared


def rubble_at_positions(state: LuxGameState, positions: np.ndarray) -> Dict[int, int]:
    rubble = {}
    for p in positions:
        rubble[pos_to_idx(p, state.env_cfg.map_size)] = state.board.rubble[p[0], p[1]]
    return rubble


@dataclass
class ActionStats:
    ACTION_NAMES = ("move", "transfer", "pickup", "dig", "self_destruct", "recharge")
    action_type: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(6, dtype=np.int32)
    )
    no_valid_action = 0
    repeat_action = 0
    move_cancelled = 0
    transfer_cancelled_no_target = 0
    transfer_cancelled_target_full = 0
    transfer_cancelled_no_resource = 0
    pickup_cancelled = 0
    build_cancelled = 0

    def stats_dict(self, prefix: str) -> Dict[str, int]:
        _dict = {
            f"{prefix}{name}": cnt
            for name, cnt in zip(self.ACTION_NAMES, self.action_type.tolist())
        }
        _dict[f"{prefix}no_valid"] = self.no_valid_action
        _dict[f"{prefix}repeat"] = self.repeat_action
        _dict[f"{prefix}move_cancelled"] = self.move_cancelled
        _dict[
            f"{prefix}transfer_cancelled_no_target"
        ] = self.transfer_cancelled_no_target
        _dict[
            f"{prefix}transfer_cancelled_target_full"
        ] = self.transfer_cancelled_target_full
        _dict[
            f"{prefix}transfer_cancelled_no_resource"
        ] = self.transfer_cancelled_no_resource
        _dict[f"{prefix}pickup_cancelled"] = self.pickup_cancelled
        _dict[f"{prefix}build_cancelled"] = self.build_cancelled
        return _dict

    def __getattr__(self, name: str):
        if name in self.ACTION_NAMES:
            return self.action_type[self.ACTION_NAMES.index(name)]
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")


class StatsTracking:
    env: LuxAI_S2
    agents: List[str]
    agent_stats: Tuple[AgentRunningStats, AgentRunningStats]
    action_stats: Tuple[ActionStats, ActionStats]

    def update(self, verify: bool) -> np.ndarray:
        per_agent_updates = np.stack(
            [
                self.agent_stats[idx].update(self.env, agent, verify)
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

    def reset(self, env: LuxAI_S2, verify: bool) -> None:
        self.env = env
        self.agents = env.agents
        self.agent_stats = (AgentRunningStats(env), AgentRunningStats(env))
        self.action_stats = (ActionStats(), ActionStats())
        self.update(verify)
