import logging
import os.path
from pathlib import Path
from typing import Any, Dict

import numpy as np
from gym.spaces import Dict as DictSpace
from gym.spaces import MultiDiscrete
from luxai_s2.state import ObservationStateDict

from rl_algo_impls.lux.actions import (
    ACTION_SIZES,
    enqueued_action_from_obs,
    to_lux_actions,
)
from rl_algo_impls.lux.early import bid_action
from rl_algo_impls.lux.kit.config import EnvConfig
from rl_algo_impls.lux.kit.kit import obs_to_game_state
from rl_algo_impls.lux.observation import observation_and_action_mask
from rl_algo_impls.lux.stats import ActionStats
from rl_algo_impls.runner.config import Config, EnvHyperparams, RunArgs
from rl_algo_impls.runner.running_utils import get_device, load_hyperparams, make_policy
from rl_algo_impls.shared.tensor_utils import batch_dict_keys
from rl_algo_impls.shared.vec_env.make_env import make_eval_env
from rl_algo_impls.wrappers.hwc_to_chw_observation import HwcToChwObservation
from rl_algo_impls.wrappers.vectorable_wrapper import find_wrapper


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        root_dir = Path(__file__).parent.parent.parent.absolute()
        self.player = player
        self.agents = ["player_0", "player_1"]
        self.player_idx = self.agents.index(player)
        self.faction = ["AlphaStrike", "MotherMars"][self.player_idx]

        self.env_cfg = env_cfg

        run_args = RunArgs(algo="ppo", env="LuxAI_S2-agent", seed=1)
        hyperparams = load_hyperparams(run_args.algo, run_args.env)
        config = Config(
            run_args,
            hyperparams,
            str(root_dir),
        )

        env = make_eval_env(
            config,
            EnvHyperparams(**config.env_hyperparams),
            override_hparams={"n_envs": 1},
        )
        device = get_device(config, env)
        self.policy = make_policy(
            config,
            env,
            device,
            **config.policy_hyperparams,
            load_path=os.path.join(root_dir, config.policy_hyperparams["load_path"]),
        ).eval()

        transpose_wrapper = find_wrapper(env, HwcToChwObservation)
        assert transpose_wrapper
        self.transpose_wrapper = transpose_wrapper

        self.map_size = env_cfg.map_size
        self.num_map_tiles = self.map_size * self.map_size
        self.action_plane_space = MultiDiscrete(ACTION_SIZES)
        self.action_space = DictSpace(
            {
                "per_position": MultiDiscrete(
                    np.array(ACTION_SIZES * self.num_map_tiles).flatten().tolist()
                ),
                "pick_position": MultiDiscrete([self.num_map_tiles]),
            }
        )
        self.action_mask_shape = {
            "per_position": (
                self.num_map_tiles,
                self.action_plane_space.nvec.sum(),
            ),
            "pick_position": (
                len(self.action_space["pick_position"].nvec),
                self.num_map_tiles,
            ),
        }

    def act(
        self, step: int, lux_obs: ObservationStateDict, remainingOverageTime: int = 60
    ) -> Dict[str, Any]:
        state = obs_to_game_state(step, self.env_cfg, lux_obs)
        enqueued_actions = {
            u_id: enqueued_action_from_obs(u["action_queue"])
            for p in self.agents
            for u_id, u in lux_obs["units"][p].items()
        }
        obs, action_mask = observation_and_action_mask(
            self.player,
            lux_obs,
            state,
            self.action_mask_shape,
            enqueued_actions,
            factory_ice_distance_buffer=0,
        )
        obs = np.expand_dims(obs, axis=0)
        obs = self.transpose_wrapper.observation(obs)
        action_mask = np.expand_dims(action_mask, axis=0)

        actions = self.policy.act(
            obs, deterministic=False, action_masks=batch_dict_keys(action_mask)
        )
        action_stats = ActionStats()
        lux_action = to_lux_actions(
            self.player,
            state,
            actions[0],
            action_mask[0],
            enqueued_actions,
            action_stats,
        )
        return lux_action

    def bid_policy(
        self, step: int, lux_obs: ObservationStateDict, remainingOverageTime: int = 60
    ) -> Dict[str, Any]:
        return bid_action(5, self.faction)
