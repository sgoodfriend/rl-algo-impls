from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import Wrapper
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import MultiDiscrete
from gymnasium.spaces import Tuple as TupleSpace
from gymnasium.vector.utils import batch_space
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict

from rl_algo_impls.lux.actions import (
    ACTION_SIZES,
    enqueued_action_from_obs,
    to_lux_actions,
)
from rl_algo_impls.lux.early import bid_action
from rl_algo_impls.lux.observation import observation_and_action_mask
from rl_algo_impls.lux.resource_distance_map import FactoryPlacementDistances
from rl_algo_impls.lux.rewards import LuxRewardWeights, from_lux_rewards
from rl_algo_impls.lux.stats import StatsTracking
from rl_algo_impls.wrappers.vector_wrapper import (
    VecEnvResetReturn,
    VecEnvStepReturn,
    VectorEnv,
    merge_info,
)


class LuxEnvGridnet(VectorEnv):
    def __init__(
        self,
        env: LuxAI_S2,
        bid_std_dev: float = 5,
        reward_weights: Optional[Dict[str, float]] = None,
        verify: bool = False,
        factory_ice_distance_buffer: Optional[int] = None,
        reset_on_done: bool = True,
    ) -> None:
        super().__init__()
        self.env = env
        self.bid_std_dev = bid_std_dev
        if reward_weights is None:
            self.reward_weights = LuxRewardWeights.default_start()
        else:
            self.reward_weights = LuxRewardWeights(**reward_weights)
        self.verify = verify
        self.factory_ice_distance_buffer = factory_ice_distance_buffer
        self.seed_rng = np.random.RandomState()
        self.reset_on_done = reset_on_done
        self.map_size = self.env.env_cfg.map_size

        self.stats = StatsTracking()

        self.num_map_tiles = self.map_size * self.map_size
        self.action_plane_space = MultiDiscrete(ACTION_SIZES)
        self.single_action_space = DictSpace(
            {
                "per_position": MultiDiscrete(
                    np.array(ACTION_SIZES * self.num_map_tiles).flatten().tolist()
                ),
                "pick_position": MultiDiscrete([self.num_map_tiles]),
            }
        )
        self.action_space = TupleSpace((self.single_action_space,) * 2)
        self.action_mask_shape = {
            "per_position": (
                self.num_map_tiles,
                self.action_plane_space.nvec.sum(),
            ),
            "pick_position": (
                len(self.single_action_space["pick_position"].nvec),
                self.num_map_tiles,
            ),
        }

        observation_sample, _ = self.reset()
        single_obs_shape = observation_sample.shape[1:]
        self.single_observation_space = Box(
            low=0,
            high=1,
            shape=single_obs_shape,
            dtype=np.float32,
        )
        self.observation_space = batch_space(self.single_observation_space, n=2)

        self._enqueued_actions: Dict[str, Optional[np.ndarray]] = {}
        self._action_mask: Optional[np.ndarray] = None

    @property
    def num_envs(self) -> int:
        return len(self.env.possible_agents)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def step(self, action: np.ndarray) -> VecEnvStepReturn:
        lux_actions = self._to_lux_actions(action)
        lux_obs, lux_rewards, terminations, truncations, info = self.env.step(
            lux_actions
        )

        self.factory_distances.record_placement(lux_actions)

        dones = {p: terminations[p] or truncations[p] for p in self.agents}
        all_done = all(dones.values())
        rewards = self._from_lux_rewards(lux_rewards, all_done, info)

        if all_done and self.reset_on_done:
            obs, _ = self.reset()
        else:
            if self.reset_on_done:
                assert not any(dones.values()), "All or none should be done"
            self._enqueued_actions = {
                u_id: enqueued_action_from_obs(u["action_queue"])
                for p in self.agents
                for u_id, u in lux_obs[p]["units"][p].items()
            }
            obs = self._from_lux_observation(lux_obs)

        return (
            obs,
            rewards,
            np.array([terminations[p] for p in self.agents]),
            np.array([truncations[p] for p in self.agents]),
            merge_info(self, [info.get(p, {}) for p in self.agents]),
        )

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> VecEnvResetReturn:
        if seed is not None:
            self.seed_rng = np.random.RandomState(seed)
        int_info = np.iinfo(np.int32)
        lux_obs, info, self.agents = reset_and_early_phase(
            self.env,
            self.bid_std_dev,
            self.seed_rng.randint(int_info.max),
            **kwargs,
        )
        self.factory_distances = FactoryPlacementDistances(self.env.state)
        self._enqueued_actions = {}
        self.stats.reset(self.env, self.verify)
        return self._from_lux_observation(lux_obs), merge_info(
            self, [info.get(p, {}) for p in self.agents]
        )

    def _from_lux_observation(
        self, lux_obs: Dict[str, ObservationStateDict]
    ) -> np.ndarray:
        observations = []
        action_masks = []
        for player in self.agents:
            obs, action_mask = observation_and_action_mask(
                player,
                lux_obs[player],
                self.env.state,
                self.action_mask_shape,
                self._enqueued_actions,
                factory_ice_distance_buffer=self.factory_ice_distance_buffer,
            )
            observations.append(obs)
            action_masks.append(action_mask)
        self._action_mask = np.stack(action_masks)
        return np.stack(observations)

    def get_action_mask(self) -> np.ndarray:
        assert self._action_mask is not None
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
        if done:
            for agent in self.agents:
                state_agent_stats = self.unwrapped.state.stats[agent]
                actions_success = state_agent_stats["action_queue_updates_success"]
                actions_total = state_agent_stats["action_queue_updates_total"]
                info[agent]["stats"] = {
                    **info[agent].get("stats", {}),
                    **{
                        "actions_success": actions_success,
                        "actions_failed": actions_total - actions_success,
                    },
                    **self.factory_distances.get_distances(agent),
                }
                if self.verify:
                    assert actions_total - actions_success == 0
        return from_lux_rewards(
            lux_rewards, done, info, self.stats, self.reward_weights, self.verify
        )


def bid_actions(agents: List[str], bid_std_dev: float) -> Dict[str, Any]:
    return {
        p: bid_action(bid_std_dev, f)
        for p, f in zip(agents, ["AlphaStrike", "MotherMars"])
    }


def reset_and_early_phase(
    env: LuxAI_S2, bid_std_dev: float, seed: Optional[int] = None, **kwargs
) -> Tuple[Dict[str, ObservationStateDict], dict, List[str]]:
    env.reset(seed=seed, **kwargs)
    agents = env.agents
    lux_obs, _, _, _, info = env.step(bid_actions(env.agents, bid_std_dev))
    return lux_obs, info, agents
