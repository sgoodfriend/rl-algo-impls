import inspect
import logging
import os
import pathlib
from multiprocessing import Pool
from typing import NamedTuple

import numpy as np

from rl_algo_impls.lux.vec_env.lux_replay_env import LuxReplayEnv
from rl_algo_impls.runner.config import EnvHyperparams
from rl_algo_impls.runner.running_utils import load_hyperparams


class Path(NamedTuple):
    replay: str
    npz: str


def replays_to_npz(
    replay_dir: str,
    npz_dir: str,
    env_id: str,
    algo: str = "acbc",
    skip_existing_files: bool = True,
    synchronous: bool = False,
) -> None:
    if not os.path.exists(npz_dir):
        os.makedirs(npz_dir)
    elif not os.path.isdir(npz_dir):
        raise ValueError(f"{npz_dir} must be a directory")

    paths = []
    for dirpath, _, filenames in os.walk(replay_dir):
        full_npz_path = pathlib.Path(dirpath).as_posix().replace(replay_dir, npz_dir)
        if not os.path.exists(full_npz_path):
            os.makedirs(full_npz_path)
        elif not os.path.isdir(full_npz_path):
            raise ValueError(f"{full_npz_path} must be a directory")
        for fname in filenames:
            basename, ext = os.path.splitext(fname)
            if ext != ".json" or not basename.isdigit():
                continue
            npz_filepath = os.path.join(full_npz_path, f"{basename}.npz")
            if skip_existing_files and os.path.exists(npz_filepath):
                continue
            paths.append(
                Path(
                    os.path.join(dirpath, fname),
                    npz_filepath,
                )
            )

    if synchronous:
        for replay_path, npz_path in paths:
            replay_file_to_npz(replay_path, npz_path, env_id, algo)
    else:
        with Pool() as pool:
            pool.starmap(
                replay_file_to_npz,
                [(p.replay, p.npz, env_id, algo) for p in paths],
            )


def replay_file_to_npz(
    replay_filepath: str,
    npz_filepath: str,
    env_id: str,
    algo: str,
) -> None:
    hparams = load_hyperparams(algo, env_id)
    env_hparams = EnvHyperparams(**hparams.env_hyperparams)
    make_kwargs = {}
    if env_hparams.make_kwargs:
        make_kwargs = {
            k: v
            for k, v in env_hparams.make_kwargs.items()
            if k in inspect.signature(LuxReplayEnv).parameters
        }
    env = LuxReplayEnv(lambda: replay_filepath, **make_kwargs)
    if not env.successful_ending:
        logging.warn(f"{replay_filepath} did not end successfully. Skipping")
        return

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
        "LuxAI_S2-v0-squnet-iDeimos",
        algo="acbc",
    )
