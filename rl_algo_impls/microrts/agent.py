import os
from pathlib import Path

from rl_algo_impls.runner.config import Config, EnvHyperparams, RunArgs
from rl_algo_impls.runner.running_utils import get_device, load_hyperparams, make_policy
from rl_algo_impls.shared.vec_env.make_env import make_eval_env

MODEL_LOAD_PATH = "saved_models/ppo-Microrts-selfplay-dc-phases-A10-S1-best"

if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent.absolute()

    run_args = RunArgs(algo="ppo", env="Microrts-agent", seed=1)
    hyperparams = load_hyperparams(run_args.algo, run_args.env)
    config = Config(run_args, hyperparams, str(root_dir))

    env = make_eval_env(config, EnvHyperparams(**config.env_hyperparams))
    device = get_device(config, env)
    policy = make_policy(
        config,
        env,
        device,
        load_path=os.path.join(root_dir, MODEL_LOAD_PATH),
        **config.policy_hyperparams,
    ).eval()

    get_action_mask = getattr(env, "get_action_mask")

    obs = env.reset()
    action_mask = get_action_mask()
    done = False
    while not done:
        act = policy.act(obs, deterministic=False, action_masks=action_mask)
        obs, _, d, _ = env.step(act)
        action_mask = get_action_mask()
        done = d[0]
