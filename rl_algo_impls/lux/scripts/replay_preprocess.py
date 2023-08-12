import json
import os
from multiprocessing import Pool
from typing import Dict, NamedTuple, Optional, Union

import numpy as np

from rl_algo_impls.lux.vec_env.lux_replay_env import LuxReplayEnv

DEFAULT_BEHAVIOR_COPY_REWARD_WEIGHTS: Dict[str, Union[float, int]] = {
    "score_vs_opponent": 1
}


class Path(NamedTuple):
    replay: str
    npz: str


def replays_to_npz(
    replay_dir: str,
    npz_dir: str,
    team_name: str,
    reward_weights: Optional[Dict[str, Union[float, int]]],
) -> None:
    if not os.path.exists(npz_dir):
        os.makedirs(npz_dir)
    elif not os.path.isdir(npz_dir):
        raise ValueError(f"{npz_dir} must be a directory")

    paths = []
    for dirpath, _, filenames in os.walk(replay_dir):
        for fname in filenames:
            basename, ext = os.path.splitext(fname)
            if ext != ".json" or not basename.isdigit():
                continue
            paths.append(
                Path(
                    os.path.join(dirpath, fname),
                    os.path.join(npz_dir, f"{basename}.npz"),
                )
            )

    with Pool() as pool:
        pool.starmap(
            replay_file_to_npz,
            [(p.replay, p.npz, team_name, reward_weights) for p in paths],
        )


def replay_file_to_npz(
    replay_filepath: str,
    npz_filepath: str,
    team_name: str,
    reward_weights: Optional[Dict[str, float]],
) -> None:
    env = LuxReplayEnv(lambda: replay_filepath, team_name, reward_weights)
    num_steps = env.state.num_steps

    obs = np.zeros(
        (num_steps,) + env.observation_space.shape, dtype=env.observation_space.dtype
    )
    # reward, done, actions are offset by one from obs
    reward = np.zeros(num_steps - 1, dtype=np.float32)
    done = np.full(num_steps - 1, False, dtype=np.bool_)
    actions = []

    # action_masks are not offset; however, don't compute mask for last observation
    action_masks = []

    obs[0] = env.reset()
    idx = 0
    d = False
    while not d:
        action_masks.append(env.get_action_mask())
        o, r, d, _ = env.step(None)
        reward[idx] = r
        done[idx] = d
        actions.append(env.last_action)
        idx += 1
        obs[idx] = o
    assert idx == obs.shape[0] - 1

    actions_np = {f"actions_{k}": np.array([a[k] for a in actions]) for k in actions[0]}
    action_masks_np = {
        f"action_mask_{k}": np.array([a[k] for a in action_masks])
        for k in action_masks[0]
    }
    np.savez_compressed(
        npz_filepath,
        **{
            "obs": obs,
            "reward": reward,
            "done": done,
            **actions_np,
            **action_masks_np,
        },
    )


if __name__ == "__main__":
    replays_to_npz(
        "data/lux/replays-deimos",
        "data/lux/npz-deimos",
        "Deimos",
        DEFAULT_BEHAVIOR_COPY_REWARD_WEIGHTS,
    )
