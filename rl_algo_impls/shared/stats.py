import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter


@dataclass
class Episode:
    score: float = 0
    length: int = 0
    info: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)


StatisticSelf = TypeVar("StatisticSelf", bound="Statistic")


@dataclass
class Statistic:
    values: np.ndarray
    round_digits: int = 2
    score_function: str = "mean-std"

    @property
    def mean(self) -> float:
        return np.mean(self.values).item()

    @property
    def std(self) -> float:
        return np.std(self.values).item()

    @property
    def min(self) -> float:
        return np.min(self.values).item()

    @property
    def max(self) -> float:
        return np.max(self.values).item()

    def sum(self) -> float:
        return np.sum(self.values).item()

    def __len__(self) -> int:
        return len(self.values)

    def score(self) -> float:
        if self.score_function == "mean-std":
            return self.mean - self.std
        elif self.score_function == "mean":
            return self.mean
        else:
            raise NotImplemented(
                f"Only mean-std and mean score_functions supported ({self.score_function})"
            )

    def _diff(self: StatisticSelf, o: StatisticSelf) -> float:
        return self.score() - o.score()

    def __gt__(self: StatisticSelf, o: StatisticSelf) -> bool:
        return self._diff(o) > 0

    def __ge__(self: StatisticSelf, o: StatisticSelf) -> bool:
        return self._diff(o) >= 0

    def __repr__(self) -> str:
        mean = round(self.mean, self.round_digits)
        if self.round_digits == 0:
            mean = int(mean)
        if self.score_function == "mean":
            return f"{mean}"

        std = round(self.std, self.round_digits)
        if self.round_digits == 0:
            std = int(std)
        return f"{mean} +/- {std}"

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
        }


EpisodesStatsSelf = TypeVar("EpisodesStatsSelf", bound="EpisodesStats")


class EpisodesStats:
    def __init__(
        self,
        episodes: Sequence[Episode],
        simple: bool = False,
        score_function: str = "mean-std",
    ) -> None:
        self.episodes = episodes
        self.simple = simple
        self.score = Statistic(
            np.array([e.score for e in episodes]), score_function=score_function
        )
        self.length = Statistic(np.array([e.length for e in episodes]), round_digits=0)
        additional_values = defaultdict(list)
        for e in self.episodes:
            if e.info:
                for k, v in e.info.items():
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            additional_values[f"{k}_{k2}"].append(v2)
                    else:
                        additional_values[k].append(v)
        self.additional_stats = {
            k: Statistic(np.array(values)) for k, values in additional_values.items()
        }
        self.score_function = score_function

    def __gt__(self: EpisodesStatsSelf, o: EpisodesStatsSelf) -> bool:
        return self.score > o.score

    def __ge__(self: EpisodesStatsSelf, o: EpisodesStatsSelf) -> bool:
        return self.score >= o.score

    def __repr__(self) -> str:
        mean = self.score.mean
        score = self.score.score()
        if mean != score:
            return f"Score: {self.score} ({round(score)}) | Length: {self.length}"
        else:
            return f"Score: {self.score} | Length: {self.length}"

    def __len__(self) -> int:
        return len(self.episodes)

    def _asdict(self) -> dict:
        return {
            "n_episodes": len(self.episodes),
            "score": self.score.to_dict(),
            "length": self.length.to_dict(),
        }

    def write_to_tensorboard(
        self, tb_writer: SummaryWriter, main_tag: str, global_step: Optional[int] = None
    ) -> None:
        stats = {"mean": self.score.mean}
        if not self.simple:
            stats.update(
                {
                    "min": self.score.min,
                    "max": self.score.max,
                    "result": self.score.score(),
                    "n_episodes": len(self.episodes),
                    "length": self.length.mean,
                }
            )
            for k, addl_stats in self.additional_stats.items():
                stats[k] = addl_stats.mean
        for name, value in stats.items():
            tb_writer.add_scalar(f"{main_tag}/{name}", value, global_step=global_step)


class EpisodeAccumulator:
    def __init__(self, num_envs: int):
        self._episodes = []
        self.current_episodes = [Episode() for _ in range(num_envs)]

    @property
    def episodes(self) -> List[Episode]:
        return self._episodes

    def step(self, reward: np.ndarray, done: np.ndarray, info: List[Dict]) -> None:
        for idx, current in enumerate(self.current_episodes):
            current.score += reward[idx]
            current.length += 1
            if done[idx]:
                self._episodes.append(current)
                self.current_episodes[idx] = Episode()
                self.on_done(idx, current, info[idx])

    def __len__(self) -> int:
        return len(self.episodes)

    def on_done(self, ep_idx: int, episode: Episode, info: Dict) -> None:
        pass

    def stats(self) -> EpisodesStats:
        return EpisodesStats(self.episodes)


def log_scalars(
    tb_writer: SummaryWriter,
    main_tag: str,
    tag_scalar_dict: Dict[str, Union[int, float]],
    global_step: int,
) -> None:
    for tag, value in tag_scalar_dict.items():
        tb_writer.add_scalar(f"{main_tag}/{tag}", value, global_step)
