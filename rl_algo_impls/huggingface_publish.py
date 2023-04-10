import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import shutil
import subprocess
import tempfile
from typing import List, Optional

import requests
import wandb.apis.public
from huggingface_hub.hf_api import HfApi, upload_folder
from huggingface_hub.repocard import metadata_save
from pyvirtualdisplay.display import Display

import wandb
from rl_algo_impls.publish.markdown_format import EvalTableData, model_card_text
from rl_algo_impls.runner.config import EnvHyperparams
from rl_algo_impls.runner.evaluate import EvalArgs, evaluate_model
from rl_algo_impls.shared.callbacks.eval_callback import evaluate
from rl_algo_impls.shared.vec_env import make_eval_env
from rl_algo_impls.wrappers.vec_episode_recorder import VecEpisodeRecorder


def publish(
    wandb_run_paths: List[str],
    wandb_report_url: str,
    huggingface_user: Optional[str] = None,
    huggingface_token: Optional[str] = None,
    virtual_display: bool = False,
) -> None:
    if virtual_display:
        display = Display(visible=False, size=(1400, 900))
        display.start()

    api = wandb.Api()
    runs = [api.run(rp) for rp in wandb_run_paths]
    algo = runs[0].config["algo"]
    hyperparam_id = runs[0].config["env"]
    evaluations = [
        evaluate_model(
            EvalArgs(
                algo,
                hyperparam_id,
                seed=r.config.get("seed", None),
                render=False,
                best=True,
                n_envs=None,
                n_episodes=10,
                no_print_returns=True,
                wandb_run_path="/".join(r.path),
            ),
            os.getcwd(),
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
        env_id = runs[0].config.get("env_id") or runs[0].config["env"]
        card_text = model_card_text(
            algo,
            env_id,
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
                env_id,
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
                                "name": env_id,
                                "type": env_id,
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
                EnvHyperparams(**config.env_hyperparams),
                override_hparams={"n_envs": 1},
                normalize_load_path=model_path,
            ),
            os.path.join(repo_dir_path, "replay"),
            max_video_length=3600,
        )
        evaluate(
            video_env,
            policy,
            1,
            deterministic=config.eval_hyperparams.get("deterministic", True),
        )

        api = HfApi()
        huggingface_user = huggingface_user or api.whoami()["name"]
        huggingface_repo = f"{huggingface_user}/{repo_name}"
        api.create_repo(
            token=huggingface_token,
            repo_id=huggingface_repo,
            private=False,
            exist_ok=True,
        )
        repo_url = upload_folder(
            repo_id=huggingface_repo,
            folder_path=repo_dir_path,
            path_in_repo="",
            commit_message=f"{algo.upper()} playing {env_id} from {github_url}/tree/{commit_hash}",
            token=huggingface_token,
            delete_patterns="*",
        )
        print(f"Pushed model to the hub: {repo_url}")


def huggingface_publish():
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
    parser.add_argument(
        "--virtual-display", action="store_true", help="Use headless virtual display"
    )
    args = parser.parse_args()
    print(args)
    publish(**vars(args))


if __name__ == "__main__":
    huggingface_publish()
