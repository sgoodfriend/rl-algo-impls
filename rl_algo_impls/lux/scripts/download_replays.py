import argparse
import json
import os
import subprocess
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import polars as pl
import requests

from rl_algo_impls.lux.scripts.replay_preprocess import replays_to_npz
from rl_algo_impls.runner.config import EnvHyperparams
from rl_algo_impls.runner.running_utils import load_hyperparams

LUX_COMPETITION_ID = 45040
REPLAY_DOWNLOAD_LIMIT = 1000
SCORE_THRESHOLD = 1700

BASE_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService"
GET_EPISODE_REPLAY_URL = os.path.join(BASE_URL, "GetEpisodeReplay")


def download_replays(
    meta_kaggle_dir: str,
    target_base_dir: str,
    team_name: str,
    after_date: str,
    score_threshold: int = SCORE_THRESHOLD,
    download_limit: int = REPLAY_DOWNLOAD_LIMIT,
    num_latest_submissions: Optional[int] = None,
) -> None:
    target_dir = os.path.join(f"{target_base_dir}-{team_name.lower()}", team_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    elif not os.path.isdir(target_dir):
        raise ValueError(f"{target_dir} must be a directory")

    team_id = get_team_id(meta_kaggle_dir, team_name)
    submission_ids = get_team_submission_ids(
        meta_kaggle_dir, team_id, after_date, score_threshold, num_latest_submissions
    )
    episode_agents_df = get_submission_episode_agents(meta_kaggle_dir, submission_ids)
    episodes_df = get_episodes(
        meta_kaggle_dir, episode_agents_df["EpisodeId"].to_list()
    )

    download_cnt = 0
    for ep_id in episodes_df.index:
        filepath = os.path.join(target_dir, f"{ep_id}.json")
        if os.path.exists(filepath):
            with open(filepath) as f:
                try:
                    json.load(f)
                    continue
                except json.decoder.JSONDecodeError:
                    print(f"File {filepath} corrupted. Redownloading")
                    os.remove(filepath)
        save_episode(ep_id, target_dir)
        time.sleep(1)
        download_cnt += 1
        print(f"{download_cnt:4}: Saved {filepath}")
        if download_cnt >= download_limit:
            print(f"Reached download limit: {download_limit}. Stopping")
            break
    else:
        print(f"Downloaded all files")


def get_team_id(meta_kaggle_dir: str, team_name: str) -> int:
    teams_pldf = pl.read_csv(os.path.join(meta_kaggle_dir, "Teams.csv"))
    teams_pldf = teams_pldf.filter(pl.col("CompetitionId") == LUX_COMPETITION_ID)
    teams_df = teams_pldf.to_pandas()
    print(f"{len(teams_df)} teams")
    selected_team = teams_df[teams_df["TeamName"] == team_name]
    team_id = selected_team["Id"].values[0]
    print(f"{team_name} team id: {team_id}")
    return team_id


def get_team_submission_ids(
    meta_kaggle_dir: str,
    team_id: int,
    after_date: str,
    score_threshold: int,
    num_latest_submissions: Optional[int],
) -> List[int]:
    subs_pldf = pl.read_csv(os.path.join(meta_kaggle_dir, "Submissions.csv"))
    subs_pldf = subs_pldf.filter(pl.col("TeamId") == team_id)
    subs_df = subs_pldf.to_pandas()
    subs_df["ScoreDate"] = pd.to_datetime(subs_df["ScoreDate"])
    print(f"{len(subs_df)} submissions by {team_id}")
    subs_df = subs_df[
        (subs_df["ScoreDate"] >= after_date)
        & (subs_df["PrivateScoreFullPrecision"] >= score_threshold)
    ]
    if num_latest_submissions:
        subs_df.sort_values(by="ScoreDate", inplace=True)
        subs_df = subs_df.tail(num_latest_submissions)
    print(f"Filtered down to {len(subs_df)} submissions")
    return subs_df["Id"].to_list()


def get_submission_episode_agents(
    meta_kaggle_dir: str, submission_ids: List[int]
) -> pd.DataFrame:
    ep_agents_pldf = pl.read_csv(os.path.join(meta_kaggle_dir, "EpisodeAgents.csv"))
    ep_agents_pldf = ep_agents_pldf.filter(pl.col("SubmissionId").is_in(submission_ids))
    ep_agents_df = ep_agents_pldf.to_pandas()
    print(f"{len(ep_agents_df)} episode agents")
    return ep_agents_df


def get_episodes(meta_kaggle_dir: str, ep_ids: List[int]) -> pd.DataFrame:
    eps_pldf = pl.read_csv(os.path.join(meta_kaggle_dir, "Episodes.csv"))
    eps_pldf = eps_pldf.filter(pl.col("Id").is_in(ep_ids))
    eps_df = eps_pldf.to_pandas().set_index("Id")
    print(f"{len(eps_df)} episodes")
    return eps_df


def save_episode(ep_id: int, target_dir: str) -> None:
    re = requests.post(GET_EPISODE_REPLAY_URL, json={"episodeId": ep_id})
    replay = re.json()
    filepath = os.path.join(target_dir, f"{ep_id}.json")
    with open(filepath, "w") as f:
        json.dump(replay, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--meta-kaggle-dir", default="data/meta-kaggle")
    parser.add_argument("-f", "--target-base-dir", default="data/lux/lux-replays")
    parser.add_argument("-e", "--env-id", default="LuxAI_S2-v0-squnet-iDeimos")
    parser.add_argument("-a", "--algo", default="acbc")
    parser.add_argument("-d", "--after-date", default="2023-04-01")
    parser.add_argument("--num-latest-submissions", default=None)
    parser.add_argument("-s", "--score-threshold", default=SCORE_THRESHOLD)
    parser.add_argument("-l", "--download-limit", default=REPLAY_DOWNLOAD_LIMIT)
    parser.add_argument("-P", "--no-preprocess", action="store_true")
    parser.add_argument("-k", "--upload_to_kaggle", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--preprocess-synchronous", action="store_true")
    parser.set_defaults(
        # upload_to_kaggle=True,
        # num_latest_submissions=3,
        # env_id="LuxAI_S2-v0-squnet-flg",
        skip_download=True,
        force_preprocess=True,
        # preprocess_synchronous=True,
    )
    args = parser.parse_args()

    hparams = load_hyperparams(args.algo, args.env_id)
    env_hparams = EnvHyperparams(**hparams.env_hyperparams)
    team_name = (env_hparams.make_kwargs or {}).get("team_name", "Deimos")

    if not args.skip_download:
        download_replays(
            args.meta_kaggle_dir,
            args.target_base_dir,
            team_name,
            args.after_date,
            score_threshold=args.score_threshold,
            download_limit=args.download_limit,
            num_latest_submissions=args.num_latest_submissions,
        )

    target_dir = f"{args.target_base_dir}-{team_name.lower()}"
    if not args.no_preprocess:
        npz_target_dir = f"{args.target_base_dir}-{team_name.lower()}-npz"
        replays_to_npz(
            target_dir,
            npz_target_dir,
            args.env_id,
            algo=args.algo,
            skip_existing_files=not args.force_preprocess,
            synchronous=args.preprocess_synchronous,
        )
        target_dir = npz_target_dir

    if args.upload_to_kaggle:
        metadata_path = os.path.join(target_dir, "dataset-metadata.json")
        if not os.path.exists(metadata_path):
            init_command = ["kaggle", "datasets", "init", "-p", target_dir]
            subprocess.run(init_command)
            assert os.path.exists(metadata_path)
            with open(metadata_path) as f:
                metadata = json.load(f)
            title = f"Lux Season 2 {team_name} Replays"
            dataset_id = f"sgoodfriend/lux-replays-{team_name.lower()}"
            if not args.no_preprocess:
                title += " npz"
                dataset_id += "-npz"
            metadata["title"] = title
            metadata["id"] = dataset_id
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            create_command = [
                "kaggle",
                "datasets",
                "create",
                "-p",
                target_dir,
                "-r",
                "tar",
            ]
            subprocess.run(create_command)
        else:
            version_command = [
                "kaggle",
                "datasets",
                "version",
                "-p",
                target_dir,
                "-m",
                "Updated data",
                "-r",
                "tar",
            ]
            subprocess.run(version_command)
