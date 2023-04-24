from dataclasses import astuple
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from gym import Wrapper
from gym.spaces import Box, MultiDiscrete
from gym.spaces import Tuple as TupleSpace
from gym.vector.utils import batch_space
from luxai_s2.actions import move_deltas
from luxai_s2.env import LuxAI_S2
from luxai_s2.map.position import Position
from luxai_s2.state import ObservationStateDict
from luxai_s2.unit import Unit
from luxai_s2.utils import my_turn_to_place_factory

from rl_algo_impls.shared.lux.actions import (
    ACTION_SIZES,
    FACTORY_ACTION_ENCODED_SIZE,
    FACTORY_DO_NOTHING_ACTION,
    action_array_from_queue,
    actions_equal,
    if_self_destruct_valid,
    is_build_heavy_valid,
    is_build_light_valid,
    is_dig_valid,
    is_recharge_valid,
    is_water_action_valid,
    max_move_repeats,
    valid_move_mask,
    valid_pickup_resource_mask,
    valid_transfer_direction_mask,
    valid_transfer_resource_mask,
)
from rl_algo_impls.shared.lux.observation import from_lux_observation
from rl_algo_impls.shared.lux.stats import StatsTracking

DEFAULT_REWARD_WEIGHTS = (
    10,  # WIN_LOSS
    0.0001,  # LICHEN_DELTA (clip to +/- 1)
    # Change in value stats
    0.01,  # ICE_GENERATION (2 for a day of water for factory, 0.2 for a heavy dig action)
    2e-3,  # ORE_GENERATION (1 for building a heavy robot, 0.04 for a heavy dig action)
    0.04,  # WATER_GENERATION (2 for a day of water for factory)
    0.01,  # METAL_GENERATION (1 for building a heavy robot
    0.0001,  # LICHEN_GENERATION
    0,  # BUILT_LIGHT
    0,  # BUILT_HEAVY
    -1,  # LOST_FACTORY
    # Current value stats
    0.02,  # FACTORIES_ALIVE
)


class LuxEnvGridnet(Wrapper):
    def __init__(
        self,
        env,
        bid_std_dev: float = 5,
        reward_weight: Sequence[float] = DEFAULT_REWARD_WEIGHTS,
    ) -> None:
        super().__init__(env)
        self.bid_std_dev = bid_std_dev
        self.reward_weight = np.array(reward_weight)
        self.max_lichen_delta = 1 / reward_weight[1] if reward_weight[1] else np.inf
        self.map_size = self.unwrapped.env_cfg.map_size

        self.stats = StatsTracking()

        self.num_map_tiles = self.map_size * self.map_size
        observation_sample = self.reset()
        single_obs_shape = observation_sample.shape[1:]
        self.single_observation_space = Box(
            low=0,
            high=1,
            shape=single_obs_shape,
            dtype=np.float32,
        )
        self.observation_space = batch_space(self.single_observation_space, n=2)
        self.action_plane_space = MultiDiscrete(ACTION_SIZES)
        self.single_action_space = MultiDiscrete(
            np.array(ACTION_SIZES * self.num_map_tiles).flatten().tolist()
        )
        self.action_space = TupleSpace((self.single_action_space,) * 2)
        self.action_mask_shape = (
            self.num_map_tiles,
            self.action_plane_space.nvec.sum(),
        )

        self._enqueued_actions: Dict[str, Optional[np.ndarray]] = {}
        self._action_mask: Optional[np.ndarray] = None

    @property
    def unwrapped(self) -> LuxAI_S2:
        unwrapped = super().unwrapped
        assert isinstance(unwrapped, LuxAI_S2)
        return unwrapped

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]],]:
        env = self.unwrapped
        lux_actions = self._to_lux_actions(action)
        lux_obs, lux_rewards, done, info = env.step(lux_actions)

        all_done = all(done.values())
        rewards = self._from_lux_rewards(lux_rewards, all_done, info)

        if all_done:
            obs = self.reset()
        else:
            assert not any(done.values()), "All or none should be done"
            self._enqueued_actions = {
                u_id: action_array_from_queue(u.action_queue)
                for p in self.agents
                for u_id, u in env.state.units[p].items()
            }
            obs = self._from_lux_observation(lux_obs)

        self._action_mask = None
        return (
            obs,
            rewards,
            np.array([done[p] for p in self.agents]),
            [info[p] for p in self.agents],
        )

    def reset(self) -> np.ndarray:
        lux_obs, self.agents = reset_and_early_phase(self.unwrapped, self.bid_std_dev)
        self._enqueued_actions = {}
        self.stats.reset(self.unwrapped)
        self._action_mask = None
        return self._from_lux_observation(lux_obs)

    def _from_lux_observation(
        self, lux_obs: Dict[str, ObservationStateDict]
    ) -> np.ndarray:
        return np.stack(
            [
                from_lux_observation(
                    self.agents,
                    idx,
                    lux_obs[player_id],
                    self.env.state,
                    self._enqueued_actions,
                )
                for idx, player_id in enumerate(self.agents)
            ]
        )

    def get_action_mask(self) -> np.ndarray:
        if self._action_mask is not None:
            return self._action_mask
        action_mask = np.full(
            (2,) + self.action_mask_shape,
            False,
            dtype=np.bool_,
        )
        env = self.unwrapped
        config = env.env_cfg
        for idx, p in enumerate(self.agents):
            for f in env.state.factories[p].values():
                action_mask[
                    idx, self._pos_to_idx(f.pos), :FACTORY_ACTION_ENCODED_SIZE
                ] = np.array(
                    [
                        is_build_light_valid(f, config),
                        is_build_heavy_valid(f, config),
                        is_water_action_valid(f, config),
                        True,  # Do nothing is always valid
                    ]
                )
            move_masks = {
                u_id: valid_move_mask(
                    u, env.state, config, self._enqueued_actions.get(u_id)
                )
                for u_id, u in env.state.units[p].items()
            }
            move_validity_map = np.zeros((self.map_size, self.map_size), dtype=np.int16)
            for u_id, valid_moves_mask in move_masks.items():
                u = env.state.units[p][u_id]
                for direction_idx, move_delta in enumerate(move_deltas):
                    if valid_moves_mask[direction_idx]:
                        move_validity_map[
                            u.pos.x + move_delta[0], u.pos.y + move_delta[1]
                        ] += 1
            for u_id, u in env.state.units[p].items():
                enqueued_action = self._enqueued_actions.get(u_id)
                move_mask = move_masks[u_id]
                transfer_direction_mask = valid_transfer_direction_mask(
                    u, env.state, config, move_mask, move_validity_map, enqueued_action
                )
                transfer_resource_mask = (
                    valid_transfer_resource_mask(u)
                    if np.any(transfer_direction_mask)
                    else np.zeros(5)
                )
                pickup_resource_mask = valid_pickup_resource_mask(
                    u, env.state, enqueued_action
                )
                valid_action_types = np.array(
                    [
                        np.any(move_mask),
                        np.any(transfer_direction_mask),
                        np.any(pickup_resource_mask),
                        is_dig_valid(u, env.state, enqueued_action),
                        if_self_destruct_valid(u, env.state, enqueued_action),
                        is_recharge_valid(u, enqueued_action),
                    ]
                )
                action_mask[
                    idx, self._pos_to_idx(u.pos), FACTORY_ACTION_ENCODED_SIZE:
                ] = np.concatenate(
                    [
                        valid_action_types,
                        move_mask,
                        transfer_direction_mask,
                        transfer_resource_mask,
                        pickup_resource_mask,
                    ]
                )
        self._action_mask = action_mask
        return action_mask

    def _pos_to_idx(self, pos: Position) -> int:
        return pos.x * self.map_size + pos.y

    def _no_valid_unit_actions(self, unit: Unit) -> bool:
        assert self._action_mask is not None
        return not np.any(
            self._action_mask[
                unit.team.team_id,
                self._pos_to_idx(unit.pos),
                FACTORY_ACTION_ENCODED_SIZE : FACTORY_ACTION_ENCODED_SIZE + 6,
            ]
        )

    def _to_lux_actions(self, actions: np.ndarray) -> Dict[str, Any]:
        env = self.unwrapped
        cfg = env.env_cfg

        lux_actions = {p: {} for p in self.agents}
        for p_idx in range(len(actions)):
            p = self.agents[p_idx]
            for f in env.state.factories[p].values():
                a = actions[p_idx, self._pos_to_idx(f.pos), 0]
                if a != FACTORY_DO_NOTHING_ACTION:
                    lux_actions[p][f.unit_id] = a
            for u in env.state.units[p].values():
                a = actions[p_idx, self._pos_to_idx(u.pos), 1:]
                if self._no_valid_unit_actions(u):
                    if cfg.verbose > 1:
                        print(f"No valid action for unit {u}")
                    self.stats.action_stats[p_idx].no_valid_action += 1
                    continue
                self.stats.action_stats[p_idx].action_type[a[0]] += 1
                if actions_equal(a, self._enqueued_actions.get(u.unit_id)):
                    self.stats.action_stats[p_idx].repeat_action += 1
                    continue

                def resource_amount(unit: Unit, idx: int) -> int:
                    if idx == 4:
                        return unit.power
                    return astuple(unit.cargo)[idx]

                repeat = cfg.max_episode_length
                if a[0] == 0:  # move
                    direction = a[1]
                    resource = 0
                    amount = 0
                    repeat = max_move_repeats(u, direction, cfg)
                elif a[0] == 1:  # transfer
                    direction = a[2]
                    resource = a[3]
                    amount = resource_amount(
                        u, resource
                    )  # TODO: This can lead to waste (especially for light robots)
                elif a[0] == 2:  # pickup
                    direction = 0
                    resource = a[4]
                    capacity = u.cargo_space if resource < 4 else u.battery_capacity
                    amount = capacity - resource_amount(u, resource)
                elif a[0] == 3:  # dig
                    direction = 0
                    resource = 0
                    amount = 0
                elif a[0] == 4:  # self-destruct
                    direction = 0
                    resource = 0
                    amount = 0
                elif a[0] == 5:  # recharge
                    direction = 0
                    resource = 0
                    amount = u.battery_capacity
                else:
                    raise ValueError(f"Unrecognized action f{a[0]}")
                lux_actions[p][u.unit_id] = [
                    np.array([a[0], direction, resource, amount, 0, repeat])
                ]
        return lux_actions

    def _from_lux_rewards(
        self, lux_rewards: Dict[str, float], done: bool, info: Dict[str, Any]
    ) -> np.ndarray:
        agents = self.agents
        player_opponent = tuple((p, opp) for p, opp in zip(agents, reversed(agents)))
        _win_loss = np.array(
            [
                (
                    1
                    if lux_rewards[p] > lux_rewards[opp]
                    else (-1 if lux_rewards[p] < lux_rewards[opp] else 0)
                )
                if done
                else 0
                for p, opp in player_opponent
            ]
        )
        _stats_delta = self.stats.update()
        _lichen_delta = np.clip(
            np.array(
                [
                    _stats_delta[p_idx][4] - _stats_delta[o_idx][4]
                    for p_idx, o_idx in zip(range(2), reversed(range(2)))
                ]
            ),
            -self.max_lichen_delta,
            self.max_lichen_delta,
        )
        raw_rewards = np.concatenate(
            [
                np.expand_dims(_win_loss, axis=-1),
                np.expand_dims(_lichen_delta, axis=-1),
                _stats_delta,
            ],
            axis=-1,
        )
        if done:
            for idx, agent in enumerate(self.agents):
                agent_stats = self.stats.agent_stats[idx]
                info[agent]["stats"] = dict(
                    zip(agent_stats.NAMES, agent_stats.stats.tolist())
                )
                state_agent_stats = self.unwrapped.state.stats[agent]
                actions_success = state_agent_stats["action_queue_updates_success"]
                actions_total = state_agent_stats["action_queue_updates_total"]
                info[agent]["stats"]["actions_success"] = actions_success
                info[agent]["stats"]["actions_failed"] = actions_total - actions_success
                info[agent]["stats"].update(
                    self.stats.action_stats[idx].stats_dict(prefix="actions_")
                )
                info[agent]["results"] = {
                    "WinLoss": _win_loss[idx],
                    "win": int(_win_loss[idx] == 1),
                    "loss": int(_win_loss[idx] == -1),
                    "score": lux_rewards[agent],
                    "score_delta": lux_rewards[agent]
                    - lux_rewards[player_opponent[idx][1]],
                }
        return np.sum(raw_rewards * self.reward_weight, axis=-1)


def reset_and_early_phase(
    env: LuxAI_S2, bid_std_dev: float
) -> Tuple[Dict[str, ObservationStateDict], List[str]]:
    env.reset()
    agents = env.agents
    env.step(bid_action(env, bid_std_dev))
    while env.state.real_env_steps < 0:
        env.step(place_factory_action(env))
    lux_obs, _, _, _ = env.step(place_initial_robot_action(env))
    return lux_obs, agents


def bid_action(env: LuxAI_S2, bid_std_dev: float) -> Dict[str, Any]:
    return {
        p: {"bid": b, "faction": f}
        for p, b, f in zip(
            env.agents,
            np.round(np.random.normal(scale=bid_std_dev, size=2)).astype(int).tolist(),
            ["AlphaStrike", "MotherMars"],
        )
    }


def place_factory_action(env: LuxAI_S2) -> Dict[str, Any]:
    player_idx_to_place = int(
        my_turn_to_place_factory(
            env.state.teams[env.agents[0]].place_first, env.state.real_env_steps
        )
    )
    p1 = env.agents[player_idx_to_place]
    p2 = env.agents[(player_idx_to_place + 1) % 2]
    own_factories = np.array([f.pos.pos for f in env.state.factories[p1].values()])
    opp_factories = np.array([f.pos.pos for f in env.state.factories[p2].values()])

    water_left = env.state.teams[p1].init_water
    metal_left = env.state.teams[p1].init_metal

    potential_spawns = np.argwhere(env.state.board.valid_spawns_mask)

    ice_tile_locations = np.argwhere(env.state.board.ice)
    ore_tile_locations = np.argwhere(env.state.board.ore)
    if env.env_cfg.verbose > 2 and (
        len(ice_tile_locations) == 0 or len(ore_tile_locations) == 0
    ):
        print(
            f"Map missing ice ({len(ice_tile_locations)}) or ore ({len(ore_tile_locations)})"
        )

    best_score = -1e6
    best_loc = potential_spawns[0]

    _rubble = env.state.board.rubble
    d_rubble = 10

    for loc in potential_spawns:
        ice_distances = np.linalg.norm(ice_tile_locations - loc, ord=1, axis=1)
        ore_distances = np.linalg.norm(ore_tile_locations - loc, ord=1, axis=1)
        closest_ice = np.min(ice_distances) if len(ice_distances) else 0
        closest_ore = np.min(ore_distances) if len(ore_distances) else 0

        min_loc = np.clip(loc - d_rubble, 0, env.env_cfg.map_size - 1)
        max_loc = np.clip(loc + d_rubble, 0, env.env_cfg.map_size - 1)
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
        p1: {
            "metal": min(env.env_cfg.INIT_WATER_METAL_PER_FACTORY, metal_left),
            "water": min(env.env_cfg.INIT_WATER_METAL_PER_FACTORY, water_left),
            "spawn": best_loc.tolist(),
        }
    }


def place_initial_robot_action(env: LuxAI_S2) -> Dict[str, Any]:
    return {p: {f: 1 for f in env.state.factories[p].keys()} for p in env.agents}
