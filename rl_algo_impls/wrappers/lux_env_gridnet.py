from dataclasses import astuple
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from gym import Wrapper
from gym.spaces import Box, MultiDiscrete
from gym.spaces import Tuple as TupleSpace
from gym.vector.utils import batch_space
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict

from rl_algo_impls.shared.lux.action_mask import get_action_mask
from rl_algo_impls.shared.lux.actions import (
    ACTION_SIZES,
    enqueued_action_from_obs,
    to_lux_actions,
)
from rl_algo_impls.shared.lux.early import bid_action, place_factory_action
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
                u_id: enqueued_action_from_obs(u["action_queue"])
                for p in self.agents
                for u_id, u in lux_obs[p]["units"][p].items()
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
        self._action_mask = np.stack(
            [
                get_action_mask(
                    player,
                    self.env.state,
                    self.action_mask_shape,
                    self._enqueued_actions,
                )
                for player in self.agents
            ]
        )
        return self._action_mask

    def _to_lux_actions(self, actions: np.ndarray) -> Dict[str, Any]:
        action_mask = self._action_mask
        assert action_mask is not None
        return {
            p: to_lux_actions(
                p,
                self.env.state,
                actions[p_idx],
                action_mask[p_idx],
                self._enqueued_actions,
                self.stats.action_stats[p_idx],
            )
            for p_idx, p in enumerate(self.agents)
        }

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


def bid_actions(agents: List[str], bid_std_dev: float) -> Dict[str, Any]:
    return {
        p: bid_action(bid_std_dev, f)
        for p, f in zip(agents, ["AlphaStrike", "MotherMars"])
    }


def place_factory_actions(env: LuxAI_S2) -> Dict[str, Any]:
    actions = {
        p: place_factory_action(env.state, env.agents, p_idx)
        for p_idx, p in enumerate(env.agents)
    }
    actions = {k: v for k, v in actions.items() if k}
    return actions


def reset_and_early_phase(
    env: LuxAI_S2, bid_std_dev: float
) -> Tuple[Dict[str, ObservationStateDict], List[str]]:
    env.reset()
    agents = env.agents
    env.step(bid_actions(env.agents, bid_std_dev))
    while env.state.real_env_steps < 0:
        env.step(place_factory_actions(env))
    lux_obs, _, _, _ = env.step(place_initial_robot_action(env))
    return lux_obs, agents


def place_initial_robot_action(env: LuxAI_S2) -> Dict[str, Any]:
    return {p: {f: 1 for f in env.state.factories[p].keys()} for p in env.agents}
