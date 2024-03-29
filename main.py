import json
import logging
import sys
from argparse import Namespace

from rl_algo_impls.lux.agent import Agent
from rl_algo_impls.lux.kit.config import EnvConfig
from rl_algo_impls.lux.kit.kit import process_action, process_obs

### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = (
    dict()
)  # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs = dict()


def agent_fn(observation, configurations):
    """main
    agent definition for kaggle submission.
    """
    global agent_dict
    step = observation.step

    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        env_cfg = EnvConfig.from_dict(configurations["env_cfg"])
        agent_dict[player] = Agent(player, env_cfg)
        agent_prev_obs[player] = dict()
        agent = agent_dict[player]

    agent = agent_dict[player]
    obs = process_obs(player, agent_prev_obs[player], step, json.loads(observation.obs))
    agent_prev_obs[player] = obs
    agent.step = step
    if step == 0:
        actions = agent.bid_policy(step, obs, remainingOverageTime)
    else:
        actions = agent.act(step, obs, remainingOverageTime)

    return process_action(actions)


if __name__ == "__main__":

    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    step = 0
    player_id = 0
    configurations = None
    i = 0
    while True:
        inputs = read_input()
        obs = json.loads(inputs)

        observation = Namespace(
            **dict(
                step=obs["step"],
                obs=json.dumps(obs["obs"]),
                remainingOverageTime=obs["remainingOverageTime"],
                player=obs["player"],
                info=obs["info"],
            )
        )
        if i == 0:
            configurations = obs["info"]["env_cfg"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=configurations))
        # send actions to engine
        print(json.dumps(actions))
