import argparse

import yaml

from rl_algo_impls.runner.running_utils import load_hyperparam_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str)
    parser.add_argument("-a", "--algo", default="appo")
    parser.set_defaults(
        env_id="LuxAI_S2-v0-j512env64-80m-ent1-lr01-mgn2-base1lc-nga-tkl5cl-3r-2a100"
    )
    args = parser.parse_args()

    hyperparam_dict = load_hyperparam_dict(args.algo, args.env_id)
    print(yaml.dump({args.env_id: hyperparam_dict}))
