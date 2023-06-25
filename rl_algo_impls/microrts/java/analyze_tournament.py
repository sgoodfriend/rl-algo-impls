import argparse
import csv
import os
from dataclasses import asdict, dataclass
from typing import Iterator, List, Sequence

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


@dataclass
class Match:
    iteration: int
    map_id: int
    ai1_id: int
    ai2_id: int
    steps: int
    winner: int
    crashed: int
    timedout: int


def read_matches(cols: Sequence[str], row_iter: Iterator[Sequence[str]]) -> List[Match]:
    matches = []
    for r in row_iter:
        if len(r) < 8:
            break
        matches.append(Match(*[int(c) for c in r]))
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory_name", nargs="?", default="tournament_34")
    args = parser.parse_args()

    filename = os.path.join(
        os.path.dirname(__file__), args.directory_name, "tournament.csv"
    )
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        iterator = iter(reader)
        for r in iterator:
            if r[0] == "AIs":
                ais = read_ais_rows(iterator)
                maps = read_maps_rows(iterator)
            if r[0] == "iteration":
                matches = read_matches(r, iterator)

    maps_by_idx = {idx: m for idx, m in enumerate(maps)}
    ais_by_idx = {idx: ai for idx, ai in enumerate(ais)}
    df = pd.DataFrame([asdict(m) for m in matches])

    ai1_wins = df[(df["ai1_id"] == 0) & (df["winner"] == 0)]
    ai1_ties = df[(df["ai1_id"] == 0) & (df["winner"] == -1)]
    player_1_points = ai1_wins.pivot_table(
        index="map_id", columns="ai2_id", aggfunc="size", fill_value=0
    ).add(
        ai1_ties.pivot_table(
            index="map_id", columns="ai2_id", aggfunc="size", fill_value=0
        )
        * 0.5,
        fill_value=0,
    )
    print(player_1_points.rename(index=maps_by_idx, columns=ais_by_idx))

    ai2_losses = df[(df["ai2_id"] == 0) & (df["winner"] == 0)]
    ai2_ties = df[(df["ai2_id"] == 0) & (df["winner"] == -1)]
    player_2_points = ai2_losses.pivot_table(
        index="map_id", columns="ai1_id", aggfunc="size", fill_value=0
    ).add(
        ai2_ties.pivot_table(
            index="map_id", columns="ai1_id", aggfunc="size", fill_value=0
        )
        * 0.5,
        fill_value=0,
    )

    print(player_2_points.rename(index=maps_by_idx, columns=ais_by_idx))

    points_table = player_1_points.subtract(player_2_points, fill_value=0)
    points_table.loc["AI Total"] = points_table.sum(axis=0)
    points_table["Map Total"] = points_table.sum(axis=1)
    print(points_table.rename(index=maps_by_idx, columns=ais_by_idx))
