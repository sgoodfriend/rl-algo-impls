import os
import pandas as pd
import wandb.apis.public
import yaml

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, TypeVar
from urllib.parse import urlparse

from rl_algo_impls.runner.evaluate import Evaluation

EvaluationRowSelf = TypeVar("EvaluationRowSelf", bound="EvaluationRow")


@dataclass
class EvaluationRow:
    algo: str
    env: str
    seed: Optional[int]
    reward_mean: float
    reward_std: float
    eval_episodes: int
    best: str
    wandb_url: str

    @staticmethod
    def data_frame(rows: List[EvaluationRowSelf]) -> pd.DataFrame:
        results = defaultdict(list)
        for r in rows:
            for k, v in asdict(r).items():
                results[k].append(v)
        return pd.DataFrame(results)


class EvalTableData(NamedTuple):
    run: wandb.apis.public.Run
    evaluation: Evaluation


def evaluation_table(table_data: Iterable[EvalTableData]) -> str:
    best_stats = sorted(
        [d.evaluation.stats for d in table_data], key=lambda r: r.score, reverse=True
    )[0]
    table_data = sorted(table_data, key=lambda d: d.evaluation.config.seed() or 0)
    rows = [
        EvaluationRow(
            config.algo,
            config.env_id,
            config.seed(),
            stats.score.mean,
            stats.score.std,
            len(stats),
            "*" if stats == best_stats else "",
            f"[wandb]({r.url})",
        )
        for (r, (_, stats, config)) in table_data
    ]
    df = EvaluationRow.data_frame(rows)
    return df.to_markdown(index=False)


def github_project_link(github_url: str) -> str:
    return f"[{urlparse(github_url).path}]({github_url})"


def header_section(algo: str, env: str, github_url: str, wandb_report_url: str) -> str:
    algo_caps = algo.upper()
    lines = [
        f"# **{algo_caps}** Agent playing **{env}**",
        f"This is a trained model of a **{algo_caps}** agent playing **{env}** using "
        f"the {github_project_link(github_url)} repo.",
        f"All models trained at this commit can be found at {wandb_report_url}.",
    ]
    return "\n\n".join(lines)


def github_tree_link(github_url: str, commit_hash: Optional[str]) -> str:
    if not commit_hash:
        return github_project_link(github_url)
    return f"[{commit_hash[:7]}]({github_url}/tree/{commit_hash})"


def results_section(
    table_data: List[EvalTableData], algo: str, github_url: str, commit_hash: str
) -> str:
    # type: ignore
    lines = [
        "## Training Results",
        f"This model was trained from {len(table_data)} trainings of **{algo.upper()}** "
        + "agents using different initial seeds. "
        + f"These agents were trained by checking out "
        + f"{github_tree_link(github_url, commit_hash)}. "
        + "The best and last models were kept from each training. "
        + "This submission has loaded the best models from each training, reevaluates "
        + "them, and selects the best model from these latest evaluations (mean - std).",
    ]
    lines.append(evaluation_table(table_data))
    return "\n\n".join(lines)


def prerequisites_section() -> str:
    return """
### Prerequisites: Weights & Biases (WandB)
Training and benchmarking assumes you have a Weights & Biases project to upload runs to.
By default training goes to a rl-algo-impls project while benchmarks go to
rl-algo-impls-benchmarks. During training and benchmarking runs, videos of the best
models and the model weights are uploaded to WandB.

Before doing anything below, you'll need to create a wandb account and run `wandb
login`.
"""


def usage_section(github_url: str, run_path: str, commit_hash: str) -> str:
    return f"""
## Usage
{urlparse(github_url).path}: {github_url}

Note: While the model state dictionary and hyperaparameters are saved, the latest
implementation could be sufficiently different to not be able to reproduce similar
results. You might need to checkout the commit the agent was trained on:
{github_tree_link(github_url, commit_hash)}.
```
# Downloads the model, sets hyperparameters, and runs agent for 3 episodes
python enjoy.py --wandb-run-path={run_path}
```

Setup hasn't been completely worked out yet, so you might be best served by using Google
Colab starting from the
[colab_enjoy.ipynb](https://github.com/sgoodfriend/rl-algo-impls/blob/main/colab_enjoy.ipynb)
notebook.
"""


def training_setion(
    github_url: str, commit_hash: str, algo: str, env: str, seed: Optional[int]
) -> str:
    return f"""
## Training
If you want the highest chance to reproduce these results, you'll want to checkout the
commit the agent was trained on: {github_tree_link(github_url, commit_hash)}. While
training is deterministic, different hardware will give different results.

```
python train.py --algo {algo} --env {env} {'--seed ' + str(seed) if seed is not None else ''}
```

Setup hasn't been completely worked out yet, so you might be best served by using Google
Colab starting from the
[colab_train.ipynb](https://github.com/sgoodfriend/rl-algo-impls/blob/main/colab_train.ipynb)
notebook.
"""


def benchmarking_section(report_url: str) -> str:
    return f"""
## Benchmarking (with Lambda Labs instance)
This and other models from {report_url} were generated by running a script on a Lambda
Labs instance. In a Lambda Labs instance terminal:
```
git clone git@github.com:sgoodfriend/rl-algo-impls.git
cd rl-algo-impls
bash ./lambda_labs/setup.sh
wandb login
bash ./lambda_labs/benchmark.sh [-a {{"ppo a2c dqn vpg"}}] [-e ENVS] [-j {{6}}] [-p {{rl-algo-impls-benchmarks}}] [-s {{"1 2 3"}}]
```

### Alternative: Google Colab Pro+
As an alternative,
[colab_benchmark.ipynb](https://github.com/sgoodfriend/rl-algo-impls/tree/main/benchmarks#:~:text=colab_benchmark.ipynb),
can be used. However, this requires a Google Colab Pro+ subscription and running across
4 separate instances because otherwise running all jobs will exceed the 24-hour limit.
"""


def hyperparams_section(run_config: Dict[str, Any]) -> str:
    return f"""
## Hyperparameters
This isn't exactly the format of hyperparams in {os.path.join("hyperparams",
run_config["algo"] + ".yml")}, but instead the Wandb Run Config. However, it's very
close and has some additional data:
```
{yaml.dump(run_config)}
```
"""


def model_card_text(
    algo: str,
    env: str,
    github_url: str,
    commit_hash: str,
    wandb_report_url: str,
    table_data: List[EvalTableData],
    best_eval: EvalTableData,
) -> str:
    run, (_, _, config) = best_eval
    run_path = "/".join(run.path)
    return "\n\n".join(
        [
            header_section(algo, env, github_url, wandb_report_url),
            results_section(table_data, algo, github_url, commit_hash),
            prerequisites_section(),
            usage_section(github_url, run_path, commit_hash),
            training_setion(github_url, commit_hash, algo, env, config.seed()),
            benchmarking_section(wandb_report_url),
            hyperparams_section(run.config),
        ]
    )
