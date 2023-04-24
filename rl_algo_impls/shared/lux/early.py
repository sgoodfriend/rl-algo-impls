from typing import Any, Dict, List

import numpy as np
from luxai_s2.utils import my_turn_to_place_factory

from rl_algo_impls.shared.lux.shared import LuxGameState, pos_to_numpy


def bid_action(bid_std_dev: float, faction: str) -> Dict[str, Any]:
    return {"bid": int(np.random.normal(scale=5)), "faction": faction}


def place_factory_action(
    state: LuxGameState, agents: List[str], player_idx: int
) -> Dict[str, Any]:
    env_cfg = state.env_cfg
    player_idx_to_place = int(
        my_turn_to_place_factory(
            state.teams[agents[0]].place_first, state.real_env_steps
        )
    )
    if player_idx_to_place != player_idx:
        return {}

    p1 = agents[player_idx_to_place]
    p2 = agents[(player_idx_to_place + 1) % 2]
    own_factories = np.array(
        [pos_to_numpy(f.pos) for f in state.factories[p1].values()]
    )
    opp_factories = np.array(
        [pos_to_numpy(f.pos) for f in state.factories[p2].values()]
    )

    water_left = state.teams[p1].init_water
    metal_left = state.teams[p1].init_metal

    potential_spawns = np.argwhere(state.board.valid_spawns_mask)

    ice_tile_locations = np.argwhere(state.board.ice)
    ore_tile_locations = np.argwhere(state.board.ore)
    if env_cfg.verbose > 2 and (
        len(ice_tile_locations) == 0 or len(ore_tile_locations) == 0
    ):
        print(
            f"Map missing ice ({len(ice_tile_locations)}) or ore ({len(ore_tile_locations)})"
        )

    best_score = -1e6
    best_loc = potential_spawns[0]

    _rubble = state.board.rubble
    d_rubble = 10

    for loc in potential_spawns:
        ice_distances = np.linalg.norm(ice_tile_locations - loc, ord=1, axis=1)
        ore_distances = np.linalg.norm(ore_tile_locations - loc, ord=1, axis=1)
        closest_ice = np.min(ice_distances) if len(ice_distances) else 0
        closest_ore = np.min(ore_distances) if len(ore_distances) else 0

        min_loc = np.clip(loc - d_rubble, 0, env_cfg.map_size - 1)
        max_loc = np.clip(loc + d_rubble, 0, env_cfg.map_size - 1)
        _rubble_neighbors = _rubble[min_loc[0] : max_loc[0], min_loc[1] : max_loc[1]]
        density_rubble = np.mean(_rubble_neighbors)

        if len(own_factories):
            own_factory_distances = np.linalg.norm(own_factories - loc, ord=1, axis=1)
            closest_own_factory = np.min(own_factory_distances)
        else:
            closest_own_factory = 0
        if len(opp_factories):
            opp_factory_distances = np.linalg.norm(opp_factories - loc, ord=1, axis=1)
            closest_opp_factory = np.min(opp_factory_distances)
        else:
            closest_opp_factory = 0

        score = (
            -10 * closest_ice
            - 0.01 * closest_ore
            - 10 * density_rubble / d_rubble
            + 0.01 * closest_opp_factory
            - 0.01 * closest_own_factory
        )

        if score > best_score:
            best_score = score
            best_loc = loc

    return {
        "metal": min(env_cfg.INIT_WATER_METAL_PER_FACTORY, metal_left),
        "water": min(env_cfg.INIT_WATER_METAL_PER_FACTORY, water_left),
        "spawn": best_loc.tolist(),
    }
