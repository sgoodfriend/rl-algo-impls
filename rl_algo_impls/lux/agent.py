import numpy as np
import os

from gym.spaces import MultiDiscrete
from luxai_s2.state import ObservationStateDict

from rl_algo_impls.runner.config import Config, EnvHyperparams, RunArgs
from rl_algo_impls.runner.running_utils import get_device, load_hyperparams, make_policy
from rl_algo_impls.shared.lux.action_mask import get_action_mask
from rl_algo_impls.shared.lux.actions import ACTION_SIZES, to_lux_actions, enqueued_action_from_obs
from rl_algo_impls.shared.lux.early import bid_action, place_factory_action
from rl_algo_impls.shared.lux.observation import from_lux_observation
from rl_algo_impls.shared.lux.stats import ActionStats
from rl_algo_impls.shared.vec_env.make_env import make_eval_env
from rl_algo_impls.lux.kit.config import EnvConfig
from rl_algo_impls.lux.kit.kit import obs_to_game_state

MODEL_LOAD_PATH = "downloaded_videos/ppo-LuxAI_S2-v0-medium-S1"


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.agents = ["player_0", "player_1"]
        self.player_idx = self.agents.index(player)
        self.faction = ["AlphaStrike", "MotherMars"][self.player_idx]

        self.env_cfg = env_cfg

        run_args = RunArgs(algo="ppo", env="LuxAI_S2-v0-medium-transfer", seed=1)
        hyperparams = load_hyperparams(run_args.algo, run_args.env)
        config = Config(
            run_args,
            hyperparams,
            os.getcwd(),
        )

        env = make_eval_env(
            config,
            EnvHyperparams(**config.env_hyperparams),
            override_hparams={"n_envs": 1},
        )
        device = get_device(config, env)
        self.policy = make_policy(config, env, device, load_path=MODEL_LOAD_PATH).eval()

        self.map_size = env_cfg.map_size
        self.num_map_tiles = self.map_size * self.map_size
        self.action_plane_space = MultiDiscrete(ACTION_SIZES)
        self.action_mask_shape = (
            self.num_map_tiles,
            self.action_plane_space.nvec.sum(),
        )

    def act(self, step: int, lux_obs: ObservationStateDict, remainingOverageTime: int 60) -> Dict[str, Any]:
        state = obs_to_game_state(step, self.env_cfg, lux_obs)
        enqueued_actions = {
            u_id: enqueued_action_from_obs(u["action_queue"])
            for p in self.agents
            for u_id, u in lux_obs["units"][p].items()
        }
        obs = np.expand_dims(from_lux_observation(self.agents, self.player_idx, lux_obs, state, enqueued_actions), axis=0)
        action_mask = np.expand_dims(get_action_mask(self.player, state, self.action_mask_shape, enqueued_actions), axis=0)
        actions = self.policy(obs, deterministic=False, actions_masks=action_mask)
        action_stats = ActionStats()
        lux_action = to_lux_actions(self.player, state, actions, action_mask, enqueued_actions, action_stats)
        return lux_action

    def bid_policy(self, step: int, lux_obs: ObservationStateDict, remainingOverageTime: int 60) -> Dict[str, Any]:
        return bid_action(5, self.faction)

    def factory_placement_policy(self, step: int, lux_obs: ObservationStateDict, remainingOverageTime: int 60) -> Dict[str, Any]:
        state = obs_to_game_state(step, self.env_cfg, lux_obs)
        return place_factory_action(state, self.agents, self.player_idx)

    