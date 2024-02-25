import argparse

import yaml

from rl_algo_impls.runner.running_utils import load_hyperparam_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str)
    parser.add_argument("-a", "--algo", default="dppo")
    parser.set_defaults(env_id="Microrts-env64-240m-ent5d1-lr8d1-05wb2lwr-a100")
    args = parser.parse_args()

    hyperparam_dict = load_hyperparam_dict(args.algo, args.env_id)
    print(yaml.dump({args.env_id: hyperparam_dict}))
