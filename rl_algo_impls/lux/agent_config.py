from typing import NamedTuple, Union


class LuxAgentConfig(NamedTuple):
    min_ice: int
    min_ore: int
    disable_unit_to_unit_transfers: bool
    enable_factory_to_digger_power_transfers: bool
    disable_cargo_pickup: bool
    enable_light_water_pickup: bool
    factory_ore_distance_buffer: Union[int, None]  # ore distance before ice distance
    factory_ice_distance_buffer: Union[int, None]
    valid_spawns_mask_ore_ice_union: bool
    init_water_constant: bool
    min_water_to_lichen: int
    # Lux-specific (not part of JuxAgentConfig)
    use_simplified_spaces: bool
