from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gym import Wrapper
from gym.spaces import Box
from gym.spaces import Dict as DictSpace
from gym.spaces import MultiDiscrete
from gym.spaces import Tuple as TupleSpace
from gym.vector.utils import batch_space
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict

from rl_algo_impls.lux.actions import (
    ACTION_SIZES,
    enqueued_action_from_obs,
    to_lux_actions,
)
from rl_algo_impls.lux.early import bid_action
from rl_algo_impls.lux.observation import observation_and_action_mask
from rl_algo_impls.lux.rewards import LuxRewardWeights
from rl_algo_impls.lux.stats import StatsTracking

MIN_SCORE = -1000


class LuxEnvGridnet(Wrapper):
    def __init__(
        self,
        env,
        bid_std_dev: float = 5,
        reward_weights: Optional[Dict[str, float]] = None,
        verify: bool = False,
    ) -> None:
        super().__init__(env)
        self.bid_std_dev = bid_std_dev
        if reward_weights is None:
            self.reward_weights = LuxRewardWeights.default_start()
        else:
            self.reward_weights = LuxRewardWeights(**reward_weights)
        self.verify = verify
        self.map_size = self.unwrapped.env_cfg.map_size

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

        observation_sample = self.reset()
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

        return (
            obs,
            rewards,
            np.array([done[p] for p in self.agents]),
            [info[p] for p in self.agents],
        )

    def reset(self) -> np.ndarray:
        lux_obs, self.agents = reset_and_early_phase(self.unwrapped, self.bid_std_dev)
        self._enqueued_actions = {}
        self.stats.reset(self.unwrapped, self.verify)
        return self._from_lux_observation(lux_obs)

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
        agents = self.agents
        player_opponent = tuple((p, opp) for p, opp in zip(agents, reversed(agents)))
        if done:
            _win_loss = np.array(
                [
                    1
                    if lux_rewards[p] > lux_rewards[opp]
                    else (-1 if lux_rewards[p] < lux_rewards[opp] else 0)
                    for p, opp in player_opponent
                ]
            )

            _max_score_delta = (
                1 / self.reward_weights.score_vs_opponent
                if not np.isclose(self.reward_weights.score_vs_opponent, 0)
                else np.inf
            )
            _score_delta = np.clip(
                np.array(
                    [lux_rewards[p] - lux_rewards[opp] for p, opp in player_opponent]
                ),
                -_max_score_delta,
                _max_score_delta,
            )
            _done_rewards = np.concatenate(
                [
                    np.expand_dims(_win_loss, axis=-1),
                    np.expand_dims(_score_delta, axis=-1),
                ],
                axis=-1,
            )
            for idx, agent in enumerate(self.agents):
                agent_stats = self.stats.agent_stats[idx]
                action_stats = self.stats.action_stats[idx]
                info[agent]["stats"] = dict(
                    zip(agent_stats.NAMES, agent_stats.stats.tolist())
                )
                state_agent_stats = self.unwrapped.state.stats[agent]
                actions_success = state_agent_stats["action_queue_updates_success"]
                actions_total = state_agent_stats["action_queue_updates_total"]
                info[agent]["stats"]["actions_success"] = actions_success
                info[agent]["stats"]["actions_failed"] = actions_total - actions_success
                info[agent]["stats"].update(action_stats.stats_dict(prefix="actions_"))
                if self.verify:
                    assert actions_total - actions_success == 0
                self_score = lux_rewards[agent]
                opp_score = lux_rewards[player_opponent[idx][1]]
                score_delta = self_score - opp_score
                score_reward = score_delta / (
                    self_score + opp_score + 1 - 2 * MIN_SCORE
                )
                assert np.sign(score_reward) == np.sign(
                    _win_loss[idx]
                ), f"score_reward {score_reward} must be same sign as winloss {_win_loss[idx]}"
                info[agent]["results"] = {
                    "WinLoss": _win_loss[idx],
                    "win": int(_win_loss[idx] == 1),
                    "loss": int(_win_loss[idx] == -1),
                    "score": self_score,
                    "score_delta": score_delta,
                    "score_reward": score_reward,
                }
        else:
            _done_rewards = np.zeros((2, 2))
        _stats_delta = self.stats.update(self.verify)
        raw_rewards = np.concatenate(
            [
                _done_rewards,
                _stats_delta,
            ],
            axis=-1,
        )
        reward_weights = np.array(self.reward_weights)
        return np.sum(raw_rewards * reward_weights, axis=-1)


def bid_actions(agents: List[str], bid_std_dev: float) -> Dict[str, Any]:
    return {
        p: bid_action(bid_std_dev, f)
        for p, f in zip(agents, ["AlphaStrike", "MotherMars"])
    }


def reset_and_early_phase(
    env: LuxAI_S2, bid_std_dev: float
) -> Tuple[Dict[str, ObservationStateDict], List[str]]:
    env.reset()
    agents = env.agents
    lux_obs, _, _, _ = env.step(bid_actions(env.agents, bid_std_dev))
    return lux_obs, agents
