import numpy as np

from dataclasses import dataclass, field
from typing import Generic, List, Optional, Type, TypeVar

from rl_algo_impls.wrappers.vectorable_wrapper import VecEnvObs


@dataclass
class Trajectory:
    obs: List[np.ndarray] = field(default_factory=list)
    act: List[np.ndarray] = field(default_factory=list)
    next_obs: Optional[np.ndarray] = None
    rew: List[float] = field(default_factory=list)
    terminated: bool = False
    v: List[float] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        next_obs: np.ndarray,
        rew: float,
        terminated: bool,
        v: float,
    ):
        self.obs.append(obs)
        self.act.append(act)
        self.next_obs = next_obs if not terminated else None
        self.rew.append(rew)
        self.terminated = terminated
        self.v.append(v)

    def __len__(self) -> int:
        return len(self.obs)


T = TypeVar("T", bound=Trajectory)


class TrajectoryAccumulator(Generic[T]):
    def __init__(self, num_envs: int, trajectory_class: Type[T] = Trajectory) -> None:
        self.num_envs = num_envs
        self.trajectory_class = trajectory_class

        self._trajectories = []
        self._current_trajectories = [trajectory_class() for _ in range(num_envs)]

    def step(
        self,
        obs: VecEnvObs,
        action: np.ndarray,
        next_obs: VecEnvObs,
        reward: np.ndarray,
        done: np.ndarray,
        val: np.ndarray,
        *args,
    ) -> None:
        assert isinstance(obs, np.ndarray)
        assert isinstance(next_obs, np.ndarray)
        for i, args in enumerate(zip(obs, action, next_obs, reward, done, val, *args)):
            trajectory = self._current_trajectories[i]
            # TODO: Eventually take advantage of terminated/truncated differentiation in
            # later versions of gym.
            trajectory.add(*args)
            if done[i]:
                self._trajectories.append(trajectory)
                self._current_trajectories[i] = self.trajectory_class()
                self.on_done(i, trajectory)

    @property
    def all_trajectories(self) -> List[T]:
        return self._trajectories + list(
            filter(lambda t: len(t), self._current_trajectories)
        )

    def n_timesteps(self) -> int:
        return sum(len(t) for t in self.all_trajectories)

    def on_done(self, env_idx: int, trajectory: T) -> None:
        pass
