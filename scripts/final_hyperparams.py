import argparse

import yaml

from rl_algo_impls.runner.running_utils import load_hyperparam_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str)
    parser.add_argument("-a", "--algo", default="ppo")
    parser.set_defaults(
        env_id="Microrts-env16-80m-ent5-lr3c-mgn2-info-rew-vf24-nga-a100"
    )
    args = parser.parse_args()

    hyperparam_dict = load_hyperparam_dict(args.algo, args.env_id)
    print(yaml.dump({args.env_id: hyperparam_dict}))
