import json
import logging
from typing import Any, Dict, NamedTuple, Optional

import numpy as np
from luxai_s2.state import ObservationStateDict

from rl_algo_impls.lux.actions import enqueued_action_from_obs
from rl_algo_impls.lux.kit.config import EnvConfig
from rl_algo_impls.lux.kit.kit import GameState, obs_to_game_state
from rl_algo_impls.lux.rewards import MIN_SCORE


class LuxState(NamedTuple):
    obs_state_dict: ObservationStateDict
    game_state: GameState
    enqueued_actions: Dict[str, Optional[np.ndarray]]
    player: str


class Step(NamedTuple):
    state: LuxState
    lux_action: Dict[str, Any]
    done: bool
    self_score: float
    opponent_score: float


class LuxReplayState:
    def __init__(self, replay_path: str, team_name: str) -> None:
        self.replay_path = replay_path
        self.team_name = team_name

    def step(self) -> Step:
        self.env_step += 1
        observation = self.replay_dict["steps"][self.env_step][0]["observation"]
        obs = json.loads(observation["obs"])
        for k in ("rubble", "lichen", "lichen_strains"):
            self._update_board(obs["board"], k)

        self._update_non_board_obs(obs)

        state = self.get_state()
        done = self.env_step == len(self.replay_dict["steps"]) - 1
        self_score = 0
        opponent_score = 0
        if done:
            for idx, r in enumerate(self.replay_dict["rewards"]):
                if r is None:
                    logging.warn(f"Player {idx} had None reward. Assuming {MIN_SCORE}")
                    r = MIN_SCORE
                if idx == self.player_idx:
                    self_score = r
                else:
                    opponent_score = r
        return Step(
            state,
            self.replay_dict["steps"][self.env_step][self.player_idx]["action"],
            done,
            self_score,
            opponent_score,
        )

    def reset(self, replay_path: str) -> LuxState:
        self.replay_path = replay_path
        with open(replay_path) as f:
            self.replay_dict = json.load(f)

        self.env_cfg = EnvConfig.from_dict(self.replay_dict["configuration"]["env_cfg"])
        self.player_idx = self.replay_dict["info"]["TeamNames"].index(self.team_name)

        self.env_step = 0
        init_observation = self.replay_dict["steps"][self.env_step][0]["observation"]
        self.map_size = init_observation["height"]

        obs = json.loads(init_observation["obs"])
        self.board = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in obs["board"].items()
        }
        self._update_non_board_obs(obs)
        return self.get_state()

    @property
    def num_steps(self) -> int:
        return len(self.replay_dict["steps"])

    def get_state(self) -> LuxState:
        obs_state_dict: ObservationStateDict = dict(
            units=self.units,
            teams=self.teams,
            factories=self.factories,
            board=self.board,
            real_env_steps=self.real_env_steps,
            global_id=self.global_id,
        )  # type: ignore
        game_state = obs_to_game_state(self.env_step, self.env_cfg, obs_state_dict)
        enqueued_actions = {
            u_id: enqueued_action_from_obs(u["action_queue"])
            for p in self.units
            for u_id, u in self.units[p].items()
        }
        return LuxState(obs_state_dict, game_state, enqueued_actions, self.player)

    def _update_board(self, update: Dict[str, Any], key: str) -> None:
        for pos, v in update[key].items():
            x, y = [int(n) for n in pos.split(",")]
            self.board[key][x, y] = v

    def _update_non_board_obs(self, obs: Dict[str, Any]) -> None:
        self.units = list_to_numpy(obs["units"])
        self.factories = list_to_numpy(obs["factories"])
        self.real_env_steps = obs["real_env_steps"]
        self.global_id = obs["global_id"]
        self.player = list(self.units)[self.player_idx]

        if self.env_step == 0:
            factories_per_team = self.board["factories_per_team"]
            assert isinstance(factories_per_team, int)
            self.teams = {
                p: {
                    "team_id": p,
                    "faction": "Null",
                    "water": factories_per_team
                    * self.env_cfg.INIT_WATER_METAL_PER_FACTORY,
                    "metal": factories_per_team
                    * self.env_cfg.INIT_WATER_METAL_PER_FACTORY,
                    "factories_to_place": factories_per_team,
                    "factory_strains": [],
                }
                for p in self.units
            }
        else:
            self.teams = obs["teams"]


def list_to_numpy(
    _dict: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    d = {}
    for player in _dict:
        d[player] = {}
    for player, units in _dict.items():
        for u_id, u_data in units.items():
            d[player][u_id] = {
                k: np.array(v) if isinstance(v, list) else v for k, v in u_data.items()
            }
    return d
