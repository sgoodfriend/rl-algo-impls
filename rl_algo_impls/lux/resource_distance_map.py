from collections import defaultdict, deque
from typing import Any, Dict

import numpy as np
from luxai_s2.actions import move_deltas

from rl_algo_impls.lux.actions import is_position_in_map
from rl_algo_impls.lux.shared import LuxEnvConfig, LuxGameState


def closest_distance_map(locations: np.ndarray, cfg: LuxEnvConfig) -> np.ndarray:
    dist_map = np.full((cfg.map_size, cfg.map_size), np.inf)
    queue = deque()
    for loc in locations:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                dloc = loc + np.array([dx, dy])
                if not is_position_in_map(dloc, cfg):
                    continue
                x, y = dloc
                dist_map[x, y] = 0
                queue.append((dloc, 0))
    directions = move_deltas[1:]
    while queue:
        loc, distance = queue.popleft()
        ndist = distance + 1
        for d in directions:
            nloc = loc + d
            if not is_position_in_map(nloc, cfg):
                continue
            nx, ny = nloc
            if ndist < dist_map[nx, ny]:
                dist_map[nx, ny] = ndist
                queue.append((nloc, ndist))
    return dist_map


def ice_distance_map(state: LuxGameState) -> np.ndarray:
    return closest_distance_map(np.argwhere(state.board.ice), state.env_cfg)


def ore_distance_map(state: LuxGameState) -> np.ndarray:
    return closest_distance_map(np.argwhere(state.board.ore), state.env_cfg)


class FactoryPlacementDistances:
    def __init__(self, state: LuxGameState) -> None:
        self.ice_dist_map = ice_distance_map(state)
        self.ore_dist_map = ore_distance_map(state)
        self.ice_distances_by_player = defaultdict(list)
        self.ore_distances_by_player = defaultdict(list)

    def record_placement(self, lux_actions: Dict[str, Any]) -> None:
        for p, action in lux_actions.items():
            spawn_pos = action.get("spawn")
            if spawn_pos:
                self.ice_distances_by_player[p].append(self.ice_dist_map[spawn_pos])
                self.ore_distances_by_player[p].append(self.ore_dist_map[spawn_pos])

    def get_distances(self, player: str) -> Dict[str, Any]:
        return {
            "factory_ice_dist": np.mean(self.ice_distances_by_player[player]),
            "factory_ore_dist": np.mean(self.ore_distances_by_player[player]),
        }
