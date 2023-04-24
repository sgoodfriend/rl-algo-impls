import os

from luxai_s2.config import EnvConfig

from rl_algo_impls.runner.config import Config, EnvHyperparams, RunArgs
from rl_algo_impls.runner.running_utils import get_device, load_hyperparams, make_policy
from rl_algo_impls.shared.vec_env.make_env import make_eval_env

MODEL_LOAD_PATH = "downloaded_videos/ppo-LuxAI_S2-v0-medium-S1"


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opponent = "player_1" if self.player == "player_0" else "player_0"

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

    def act(self, step: int, obs, remainingOverageTime: int 60):
        