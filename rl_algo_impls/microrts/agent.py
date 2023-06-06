import os
import sys
from pathlib import Path

file_path = os.path.abspath(Path(__file__))
root_dir = str(Path(file_path).parent.parent.parent.absolute())
sys.path.append(root_dir)


from rl_algo_impls.microrts.vec_env.microrts_socket_env import set_connection_info
from rl_algo_impls.runner.config import Config, EnvHyperparams, RunArgs
from rl_algo_impls.runner.running_utils import get_device, load_hyperparams, make_policy
from rl_algo_impls.shared.vec_env.make_env import make_eval_env

MODEL_LOAD_PATH = "saved_models/ppo-Microrts-selfplay-dc-phases-A10-S1-best"

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        set_connection_info(int(sys.argv[1]), sys.argv[2] == "true")

    run_args = RunArgs(algo="ppo", env="Microrts-agent", seed=1)
    hyperparams = load_hyperparams(run_args.algo, run_args.env)
    config = Config(run_args, hyperparams, root_dir)

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
    # Runs forever. Java process expected to terminate on own.
    while True:
        act = policy.act(obs, deterministic=False, action_masks=action_mask)
        obs, _, d, _ = env.step(act)

        if d[0]:
            obs = env.reset()

        action_mask = get_action_mask()
