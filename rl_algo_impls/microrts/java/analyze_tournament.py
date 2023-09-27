import argparse
import csv
import os
import re
from io import StringIO
from typing import Iterator, List, Sequence

import numpy as np
import pandas as pd


def read_ais_rows(row_iter: Iterator[Sequence[str]]) -> List[str]:
    ais = []
    for r in row_iter:
        if len(r) > 1:
            ais.append(r[1])
        else:
            break
    return ais


def read_maps_rows(row_iter: Iterator[Sequence[str]]) -> List[str]:
    maps = []
    for r in row_iter:
        if len(r) > 1 and r[0] == "":
            maps.append(r[1])
        else:
            break
    return maps


def read_matches(cols: Sequence[str], row_iter: Iterator[Sequence[str]]) -> List[str]:
    matches = ["\t".join(cols)]
    for r in row_iter:
        if len(r) < 8:
            break
        matches.append("\t".join(r))
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_filepath",
        nargs="?",
        default=os.path.expanduser("~/Desktop/tournament.csv"),
    )
    parser.add_argument("out_filepath", nargs="?")
    parser.add_argument("-m", "--markdown", action="store_true")
    args = parser.parse_args()

    with open(args.in_filepath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        iterator = iter(reader)
        for r in iterator:
            if r[0] == "AIs":
                ais = read_ais_rows(iterator)
                maps = read_maps_rows(iterator)
            if r[0] == "iteration":
                matches = read_matches(r, iterator)

    maps_by_idx = {
        idx: os.path.splitext(os.path.basename(m))[0] for idx, m in enumerate(maps)
    }
    ais_by_idx = {idx: re.sub(r"\(.*\)", "", ai) for idx, ai in enumerate(ais)}
    df = pd.read_csv(StringIO("\n".join(matches)), delimiter="\t")

    ai1_wins = df[(df["ai1"] == 0) & (df["winner"] == 0)]
    ai1_ties = df[(df["ai1"] == 0) & (df["winner"] == -1)]
    player_1_points = ai1_wins.pivot_table(
        index="map", columns="ai2", aggfunc="size", fill_value=0
    ).add(
        ai1_ties.pivot_table(index="map", columns="ai2", aggfunc="size", fill_value=0)
        * 0.5,
        fill_value=0,
    )
    print(player_1_points.rename(index=maps_by_idx, columns=ais_by_idx))

    ai2_losses = df[(df["ai2"] == 0) & (df["winner"] == 0)]
    ai2_ties = df[(df["ai2"] == 0) & (df["winner"] == -1)]
    player_2_points = (
        ai2_losses.pivot_table(index="map", columns="ai1", aggfunc="size", fill_value=0)
        .add(
            ai2_ties.pivot_table(
                index="map", columns="ai1", aggfunc="size", fill_value=0
            )
            * 0.5,
            fill_value=0,
        )
        .fillna(0)
    )

    print(player_2_points.rename(index=maps_by_idx, columns=ais_by_idx))

    points_table = player_1_points.subtract(player_2_points, fill_value=0)
    points_table.loc["AI Total"] = points_table.sum(axis=0)
    points_table["Map Total"] = points_table.sum(axis=1)
    print(points_table.rename(index=maps_by_idx, columns=ais_by_idx))

    df["ai0time"] = df.apply(
        lambda r: r["ai1time"] if r["ai1"] == 0 else r["ai2time"], axis=1
    )
    df["ai0over"] = df.apply(
        lambda r: r["ai1over"] if r["ai1"] == 0 else r["ai2over"],
        axis=1,
    )
    execution_time = df.groupby("map")[["ai0time", "ai0over", "time"]].mean()
    execution_time["over%"] = 100 * execution_time["ai0over"] / execution_time["time"]
    execution_time.drop(["ai0over", "time"], axis=1, inplace=True)
    execution_time["ai0time"] = execution_time["ai0time"].round(1)
    execution_time["over%"] = execution_time["over%"].round(3)

    timedout = (
        df[
            (df["ai1"] == 0) & (df["timedout"] == 0)
            | (df["ai2"] == 0) & (df["timedout"] == 1)
        ]
        .groupby("map")
        .size()
    )
    execution_time["timedout%"] = (
        timedout.div(df.groupby("map").size()).replace(np.nan, 0) * 100
    )
    print(execution_time.rename(index=maps_by_idx))

    df["score"] = 0
    df["winrate"] = 0
    df["opponent"] = 0
    df.loc[(df["ai1"] == 0) & (df["winner"] == 0), "score"] = 1
    df.loc[(df["ai1"] == 0) & (df["winner"] == 1), "score"] = -1
    df.loc[(df["ai2"] == 0) & (df["winner"] == 1), "score"] = 1
    df.loc[(df["ai2"] == 0) & (df["winner"] == 0), "score"] = -1
    df.loc[(df["ai1"] == 0) & (df["winner"] == 0), "winrate"] = 100
    df.loc[(df["ai2"] == 0) & (df["winner"] == 1), "winrate"] = 100
    df.loc[(df["winner"] == -1), "winrate"] = 50
    df.loc[df["ai1"] == 0, "opponent"] = df["ai2"]
    df.loc[df["ai2"] == 0, "opponent"] = df["ai1"]
    score_table = df.pivot_table(
        index="map", columns="opponent", values="score", aggfunc="mean", fill_value=0
    )
    score_table.loc["AI Total"] = score_table.mean(axis=0)
    score_table["Map Total"] = score_table.mean(axis=1)
    print(score_table.rename(index=maps_by_idx, columns=ais_by_idx).round(2))

    winrate_table = df.pivot_table(
        index="map", columns="opponent", values="winrate", aggfunc="mean", fill_value=0
    )
    winrate_table.loc["AI Total"] = winrate_table.mean(axis=0)
    winrate_table["Map Total"] = winrate_table.mean(axis=1)
    print(winrate_table.rename(index=maps_by_idx, columns=ais_by_idx).round(0))

    if args.out_filepath:
        filepath = os.path.expanduser(args.out_filepath)
        format_df = (
            lambda df: df.to_markdown(None) if args.markdown else df.to_csv(None)
        )
        with open(filepath, "w") as f:
            f.write(
                "\n".join(
                    [
                        "RAISocketAI Player 1 Wins",
                        format_df(
                            player_1_points.rename(
                                index=maps_by_idx, columns=ais_by_idx
                            )
                        ),
                        "RAISocketAI Player 2 Losses",
                        format_df(
                            player_2_points.rename(
                                index=maps_by_idx, columns=ais_by_idx
                            )
                        ),
                        "RAISocketAI Point Differential",
                        format_df(
                            points_table.rename(index=maps_by_idx, columns=ais_by_idx)
                        ),
                        "RAISocketAI Average Execution Time And Over 100ms",
                        format_df(execution_time.rename(index=maps_by_idx)),
                        "RAISocketAI WinLoss",
                        format_df(
                            score_table.rename(
                                index=maps_by_idx, columns=ais_by_idx
                            ).round(2)
                        ),
                        "RAISocketAI WinRate",
                        format_df(
                            winrate_table.rename(
                                index=maps_by_idx, columns=ais_by_idx
                            ).round(0)
                        ),
                    ]
                )
            )
