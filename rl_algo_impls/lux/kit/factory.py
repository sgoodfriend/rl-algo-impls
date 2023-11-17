from dataclasses import dataclass

import numpy as np
from luxai_s2.factory import compute_water_info

from rl_algo_impls.lux.kit.cargo import UnitCargo
from rl_algo_impls.lux.kit.config import EnvConfig
from rl_algo_impls.lux.kit.board import Board


@dataclass
class Factory:
    team_id: int
    unit_id: str
    strain_id: int
    power: int
    cargo: UnitCargo
    pos: np.ndarray
    # lichen tiles connected to this factory
    # lichen_tiles: np.ndarray
    env_cfg: EnvConfig

    def build_heavy_metal_cost(self, game_state):
        unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
        return unit_cfg.METAL_COST

    def build_heavy_power_cost(self, game_state):
        unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
        return unit_cfg.POWER_COST

    def can_build_heavy(self, game_state):
        return self.power >= self.build_heavy_power_cost(
            game_state
        ) and self.cargo.metal >= self.build_heavy_metal_cost(game_state)

    def build_heavy(self):
        return 1

    def build_light_metal_cost(self, game_state):
        unit_cfg = self.env_cfg.ROBOTS["LIGHT"]
        return unit_cfg.METAL_COST

    def build_light_power_cost(self, game_state):
        unit_cfg = self.env_cfg.ROBOTS["LIGHT"]
        return unit_cfg.POWER_COST

    def can_build_light(self, game_state):
        return self.power >= self.build_light_power_cost(
            game_state
        ) and self.cargo.metal >= self.build_light_metal_cost(game_state)

    def build_light(self):
        return 0

    def water_cost(self):
        """
        Water required to perform water action
        """
        return int(
            np.ceil(
                len(self.grow_lichen_positions)
                / self.env_cfg.LICHEN_WATERING_COST_FACTOR
            )
        )

    def can_water(self, game_state):
        return self.cargo.water >= self.water_cost()

    def water(self):
        return 2

    @property
    def pos_slice(self):
        return slice(self.pos[0] - 1, self.pos[0] + 2), slice(
            self.pos[1] - 1, self.pos[1] + 2
        )

    def cache_water_info(self, board: Board):
        forbidden = (
            (board.rubble > 0)
            | (board.factory_occupancy_map != -1)
            | (board.ice > 0)
            | (board.ore > 0)
        )
        deltas = [
            np.array([0, -2]),
            np.array([-1, -2]),
            np.array([1, -2]),
            np.array([0, 2]),
            np.array([-1, 2]),
            np.array([1, 2]),
            np.array([2, 0]),
            np.array([2, -1]),
            np.array([2, 1]),
            np.array([-2, 0]),
            np.array([-2, -1]),
            np.array([-2, 1]),
        ]
        init_arr = np.stack(deltas) + self.pos
        (
            self.grow_lichen_positions,
            self.connected_lichen_positions,
        ) = compute_water_info(
            init_arr,
            self.env_cfg.MIN_LICHEN_TO_SPREAD,
            board.lichen,
            board.lichen_strains,
            board.factory_occupancy_map,
            self.strain_id,
            forbidden,
        )
