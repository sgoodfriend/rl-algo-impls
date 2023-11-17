from typing import NamedTuple, Optional, Type, TypeVar, Union

LuxAgentConfigSelf = TypeVar("LuxAgentConfigSelf", bound="LuxAgentConfig")


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

    @classmethod
    def from_kwargs(
        cls: Type[LuxAgentConfigSelf],
        factory_ore_distance_buffer: Optional[int] = None,
        factory_ice_distance_buffer: Optional[int] = None,
        valid_spawns_mask_ore_ice_union: bool = False,
        use_simplified_spaces: bool = False,
        min_ice: int = 1,
        min_ore: int = 1,
        disable_unit_to_unit_transfers: bool = False,
        enable_factory_to_digger_power_transfers: bool = False,
        disable_cargo_pickup: bool = False,
        enable_light_water_pickup: bool = False,
        init_water_constant: bool = False,
        min_water_to_lichen: int = 1000,
        **kwargs,
    ) -> LuxAgentConfigSelf:
        return cls(
            min_ice=min_ice,
            min_ore=min_ore,
            disable_unit_to_unit_transfers=disable_unit_to_unit_transfers,
            enable_factory_to_digger_power_transfers=enable_factory_to_digger_power_transfers,
            disable_cargo_pickup=disable_cargo_pickup,
            enable_light_water_pickup=enable_light_water_pickup,
            factory_ore_distance_buffer=factory_ore_distance_buffer,
            factory_ice_distance_buffer=factory_ice_distance_buffer,
            valid_spawns_mask_ore_ice_union=valid_spawns_mask_ore_ice_union,
            init_water_constant=init_water_constant,
            min_water_to_lichen=min_water_to_lichen,
            use_simplified_spaces=use_simplified_spaces,
        )
