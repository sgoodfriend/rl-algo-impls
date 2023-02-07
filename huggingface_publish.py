import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import requests
import shutil
import subprocess
import tempfile
import wandb
import wandb.apis.public

from typing import List, Optional

from huggingface_hub.hf_api import HfApi, upload_folder
from huggingface_hub.repocard import metadata_save
from publish.markdown_format import EvalTableData, model_card_text
from runner.evaluate import EvalArgs, evaluate_model
from runner.env import make_eval_env
from shared.callbacks.eval_callback import evaluate
from wrappers.vec_episode_recorder import VecEpisodeRecorder


def publish(
    wandb_run_paths: List[str],
    wandb_report_url: str,
    huggingface_user: Optional[str] = None,
    huggingface_token: Optional[str] = None,
) -> None:
    api = wandb.Api()
    runs = [api.run(rp) for rp in wandb_run_paths]
    algo = runs[0].config["algo"]
    env = runs[0].config["env"]
    evaluations = [
        evaluate_model(
            EvalArgs(
                algo,
                env,
                seed=r.config.get("seed", None),
                render=False,
                best=True,
                n_envs=None,
                n_episodes=10,
                no_print_returns=True,
                wandb_run_path="/".join(r.path),
            ),
            os.path.dirname(__file__),
        )
        for r in runs
    ]
    run_metadata = requests.get(runs[0].file("wandb-metadata.json").url).json()
    table_data = list(EvalTableData(r, e) for r, e in zip(runs, evaluations))
    best_eval = sorted(
        table_data, key=lambda d: d.evaluation.stats.score, reverse=True
    )[0]

    with tempfile.TemporaryDirectory() as tmpdirname:
        _, (policy, stats, config) = best_eval

        repo_name = config.model_name(include_seed=False)
        repo_dir_path = os.path.join(tmpdirname, repo_name)
        # Locally clone this repo to a temp directory
        subprocess.run(["git", "clone", ".", repo_dir_path])
        shutil.rmtree(os.path.join(repo_dir_path, ".git"))
        model_path = config.model_dir_path(best=True, downloaded=True)
        shutil.copytree(
            model_path,
            os.path.join(
                repo_dir_path, "saved_models", config.model_dir_name(best=True)
            ),
        )

        github_url = "https://github.com/sgoodfriend/rl-algo-impls"
        commit_hash = run_metadata.get("git", {}).get("commit", None)
        card_text = model_card_text(
            algo,
            env,
            github_url,
            commit_hash,
            wandb_report_url,
            table_data,
            best_eval,
        )
        readme_filepath = os.path.join(repo_dir_path, "README.md")
        os.remove(readme_filepath)
        with open(readme_filepath, "w") as f:
            f.write(card_text)

        metadata = {
            "library_name": "rl-algo-impls",
            "tags": [
                env,
                algo,
                "deep-reinforcement-learning",
                "reinforcement-learning",
            ],
            "model-index": [
                {
                    "name": algo,
                    "results": [
                        {
                            "metrics": [
                                {
                                    "type": "mean_reward",
                                    "value": str(stats.score),
                                    "name": "mean_reward",
                                }
                            ],
                            "task": {
                                "type": "reinforcement-learning",
                                "name": "reinforcement-learning",
                            },
                            "dataset": {
                                "name": env,
                                "type": env,
                            },
                        }
                    ],
                }
            ],
        }
        metadata_save(readme_filepath, metadata)

        video_env = VecEpisodeRecorder(
            make_eval_env(
                config,
                override_n_envs=1,
                normalize_load_path=model_path,
                **config.env_hyperparams,
            ),
            os.path.join(repo_dir_path, "replay"),
            max_video_length=3600,
        )
        evaluate(
            video_env,
            policy,
            1,
            deterministic=config.eval_params.get("deterministic", True),
        )

        api = HfApi()
        huggingface_user = huggingface_user or api.whoami()["name"]
        huggingface_repo = f"{huggingface_user}/{repo_name}"
        api.create_repo(
            token=huggingface_token,
            repo_id=huggingface_repo,
            private=True,
            exist_ok=True,
        )
        repo_url = upload_folder(
            repo_id=huggingface_repo,
            folder_path=repo_dir_path,
            path_in_repo="",
            commit_message=f"{algo.upper()} playing {env} from {github_url}/tree/{commit_hash}",
            token=huggingface_token,
        )
        print(f"Pushed model to the hub: {repo_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb-run-paths",
        type=str,
        nargs="+",
        help="Run paths of the form entity/project/run_id",
    )
    parser.add_argument("--wandb-report-url", type=str, help="Link to WandB report")
    parser.add_argument(
        "--huggingface-user",
        type=str,
        help="Huggingface user or team to upload model cards",
        default=None,
    )
    args = parser.parse_args()
    print(args)
    publish(**vars(args))
