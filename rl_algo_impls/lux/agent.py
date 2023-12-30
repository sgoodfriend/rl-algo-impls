import os.path
from pathlib import Path
from typing import Any, Dict

import numpy as np
from luxai_s2.state import ObservationStateDict

from rl_algo_impls.lux.actions import enqueued_action_from_obs, to_lux_actions
from rl_algo_impls.lux.early import bid_action
from rl_algo_impls.lux.kit.config import EnvConfig
from rl_algo_impls.lux.kit.kit import obs_to_game_state
from rl_algo_impls.lux.observation import observation_and_action_mask
from rl_algo_impls.lux.stats import ActionStats
from rl_algo_impls.lux.vec_env.lux_agent_env import LuxAgentEnv
from rl_algo_impls.runner.config import Config, EnvHyperparams, RunArgs
from rl_algo_impls.runner.running_utils import (
    get_device,
    load_hyperparams,
    make_eval_policy,
)
from rl_algo_impls.shared.data_store.synchronous_data_store_accessor import (
    SynchronousDataStoreAccessor,
)
from rl_algo_impls.shared.tensor_utils import batch_dict_keys
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces
from rl_algo_impls.shared.vec_env.make_env import get_eval_env_hyperparams


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

        env_hparams = get_eval_env_hyperparams(
            config,
            EnvHyperparams(**config.env_hyperparams),
            override_hparams={"n_envs": 1},
        )
        env_make_kwargs = env_hparams.make_kwargs or {}
        self.env = LuxAgentEnv(env_hparams.n_envs, env_cfg, **env_make_kwargs)
        device = get_device(config, EnvSpaces.from_vec_env(self.env))
        config.policy_hyperparams["load_path"] = os.path.join(
            root_dir, config.policy_hyperparams["load_path"]
        )
        self.policy = make_eval_policy(
            config,
            EnvSpaces.from_vec_env(self.env),
            device,
            SynchronousDataStoreAccessor(),
            **config.policy_hyperparams,
        ).eval()

    def act(
        self, step: int, lux_obs: ObservationStateDict, remainingOverageTime: int = 60
    ) -> Dict[str, Any]:
        state = obs_to_game_state(step, self.env_cfg, lux_obs)
        enqueued_actions = {
            u_id: enqueued_action_from_obs(
                u["action_queue"], self.env.agent_cfg.use_simplified_spaces
            )
            for p in self.agents
            for u_id, u in lux_obs["units"][p].items()
        }
        obs, action_mask = observation_and_action_mask(
            self.player,
            lux_obs,
            state,
            self.env.action_mask_shape,
            enqueued_actions,
            self.env.agent_cfg,
        )
        obs = np.expand_dims(obs, axis=0)
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
            self.env.agent_cfg,
        )
        return lux_action

    def bid_policy(
        self, step: int, lux_obs: ObservationStateDict, remainingOverageTime: int = 60
    ) -> Dict[str, Any]:
        return bid_action(self.env.bid_std_dev, self.faction)
