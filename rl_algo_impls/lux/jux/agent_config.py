from typing import NamedTuple

from rl_algo_impls.lux.jux.jax_init import jax_init

jax_init()

from jux.stats import Stats as JuxStats


class JuxAgentConfig(NamedTuple):
    # Lux and Jux shared
    min_ice: int
    min_ore: int
    disable_unit_to_unit_transfers: bool
    enable_factory_to_digger_power_transfers: bool
    disable_cargo_pickup: bool
    enable_light_water_pickup: bool
    factory_ore_distance_buffer: int  # ore distance before ice distance
    factory_ice_distance_buffer: int
    valid_spawns_mask_ore_ice_union: bool
    init_water_constant: bool
    min_water_to_lichen: int
    # Jux-specific
    relative_stats_eps: JuxStats
    use_difference_ratio: bool
