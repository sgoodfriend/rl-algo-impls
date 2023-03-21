import argparse
import itertools
import numpy as np
import pandas as pd
import wandb
import wandb.apis.public

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, TypeVar

from rl_algo_impls.benchmark_publish import RunGroup


@dataclass
class Comparison:
    control_values: List[float]
    experiment_values: List[float]

    def mean_diff_percentage(self) -> float:
        return self._diff_percentage(
            np.mean(self.control_values).item(), np.mean(self.experiment_values).item()
        )

    def median_diff_percentage(self) -> float:
        return self._diff_percentage(
            np.median(self.control_values).item(),
            np.median(self.experiment_values).item(),
        )

    def _diff_percentage(self, c: float, e: float) -> float:
        if c == e:
            return 0
        elif c == 0:
            return float("inf") if e > 0 else float("-inf")
        return 100 * (e - c) / c

    def score(self) -> float:
        return (
            np.sum(
                np.sign((self.mean_diff_percentage(), self.median_diff_percentage()))
            ).item()
            / 2
        )


RunGroupRunsSelf = TypeVar("RunGroupRunsSelf", bound="RunGroupRuns")


class RunGroupRuns:
    def __init__(
        self,
        run_group: RunGroup,
        control: List[str],
        experiment: List[str],
        summary_stats: List[str] = ["best_eval", "eval", "train_rolling"],
        summary_metrics: List[str] = ["mean", "result"],
    ) -> None:
        self.algo = run_group.algo
        self.env = run_group.env_id
        self.control = set(control)
        self.experiment = set(experiment)

        self.summary_stats = summary_stats
        self.summary_metrics = summary_metrics

        self.control_runs = []
        self.experiment_runs = []

    def add_run(self, run: wandb.apis.public.Run) -> None:
        wandb_tags = set(run.config.get("wandb_tags", []))
        if self.control & wandb_tags:
            self.control_runs.append(run)
        elif self.experiment & wandb_tags:
            self.experiment_runs.append(run)

    def comparisons_by_metric(self) -> Dict[str, Comparison]:
        c_by_m = {}
        for metric in (
            f"{s}/{m}"
            for s, m in itertools.product(self.summary_stats, self.summary_metrics)
        ):
            c_by_m[metric] = Comparison(
                [c.summary[metric] for c in self.control_runs],
                [e.summary[metric] for e in self.experiment_runs],
            )
        return c_by_m

    @staticmethod
    def data_frame(rows: Iterable[RunGroupRunsSelf]) -> pd.DataFrame:
        results = defaultdict(list)
        for r in rows:
            if not r.control_runs or not r.experiment_runs:
                continue
            results["algo"].append(r.algo)
            results["env"].append(r.env)
            results["control"].append(r.control)
            results["expierment"].append(r.experiment)
            c_by_m = r.comparisons_by_metric()
            results["score"].append(
                sum(m.score() for m in c_by_m.values()) / len(c_by_m)
            )
            for m, c in c_by_m.items():
                results[f"{m}_mean"].append(c.mean_diff_percentage())
                results[f"{m}_median"].append(c.median_diff_percentage())
        return pd.DataFrame(results)


def compare_runs() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--wandb-project-name",
        type=str,
        default="rl-algo-impls-benchmarks",
        help="WandB project name to load runs from",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB team. None uses default entity",
    )
    parser.add_argument(
        "-n",
        "--wandb-hostname-tag",
        type=str,
        nargs="*",
        help="WandB tags for hostname (i.e. host_192-9-145-26)",
    )
    parser.add_argument(
        "-c",
        "--wandb-control-tag",
        type=str,
        nargs="+",
        help="WandB tag for control commit (i.e. benchmark_5598ebc)",
    )
    parser.add_argument(
        "-e",
        "--wandb-experiment-tag",
        type=str,
        nargs="+",
        help="WandB tag for experiment commit (i.e. benchmark_5540e1f)",
    )
    parser.add_argument(
        "--envs",
        type=str,
        nargs="*",
        help="If specified, only compare these envs",
    )
    parser.add_argument(
        "--exclude-envs",
        type=str,
        nargs="*",
        help="Environments to exclude from comparison",
    )
    # parser.set_defaults(
    #     wandb_hostname_tag=["host_150-230-44-105", "host_155-248-214-128"],
    #     wandb_control_tag=["benchmark_fbc943f"],
    #     wandb_experiment_tag=["benchmark_f59bf74"],
    #     exclude_envs=[],
    # )
    args = parser.parse_args()
    print(args)

    api = wandb.Api()
    all_runs = api.runs(
        path=f"{args.wandb_entity or api.default_entity}/{args.wandb_project_name}",
        order="+created_at",
    )

    runs_by_run_group: Dict[RunGroup, RunGroupRuns] = {}
    wandb_hostname_tags = set(args.wandb_hostname_tag)
    for r in all_runs:
        if r.state != "finished":
            continue
        wandb_tags = set(r.config.get("wandb_tags", []))
        if not wandb_tags or not wandb_hostname_tags & wandb_tags:
            continue
        rg = RunGroup(r.config["algo"], r.config.get("env_id") or r.config["env"])
        if args.exclude_envs and rg.env_id in args.exclude_envs:
            continue
        if args.envs and rg.env_id not in args.envs:
            continue
        if rg not in runs_by_run_group:
            runs_by_run_group[rg] = RunGroupRuns(
                rg,
                args.wandb_control_tag,
                args.wandb_experiment_tag,
            )
        runs_by_run_group[rg].add_run(r)
    df = RunGroupRuns.data_frame(runs_by_run_group.values()).round(decimals=2)
    print(f"**Total Score: {sum(df.score)}**")
    df.loc["mean"] = df.mean(numeric_only=True)
    print(df.to_markdown())


if __name__ == "__main__":
    compare_runs()
