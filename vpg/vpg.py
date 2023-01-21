import numpy as np
import torch
import torch.nn as nn

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter
from typing import List, Optional, Sequence, NamedTuple

from shared.algorithm import Algorithm
from shared.callbacks.callback import Callback
from shared.stats import EpisodeAccumulator, EpisodesStats
from shared.trajectory import Trajectory
from shared.utils import discounted_cumsum
from vpg.policy import VPGActorCritic


class TrajectoryAccumulator:
    def __init__(self, num_envs: int, goal_steps: int):
        self.num_envs = num_envs

        self.trajectories = []
        self.current_trajectories = [Trajectory() for _ in range(num_envs)]

        self.steps_per_env = int(np.ceil(goal_steps / num_envs))
        self.step_idx = 0
        self.envs_done: set[int] = set()

        self._stats = EpisodeAccumulator(num_envs)

    def step(
        self,
        obs: VecEnvObs,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        val: np.ndarray,
    ) -> None:
        assert isinstance(obs, np.ndarray)
        self.step_idx += 1
        for i, trajectory in enumerate(self.current_trajectories):
            trajectory.add(obs[i], action[i], reward[i], val[i])
            if done[i]:
                # TODO: Eventually take advantage of terminated/truncated
                # differentiation in later versions of gym.
                trajectory.terminated = True
                self.trajectories.append(trajectory)
                self.current_trajectories[i] = Trajectory()
                if self.step_idx >= self.steps_per_env:
                    self.envs_done.add(i)
        self._stats.step(reward, done)

    def is_done(self) -> bool:
        return len(self.envs_done) == self.num_envs

    def n_timesteps(self) -> int:
        return np.sum([len(t) for t in self.trajectories]).item()

    def stats(self) -> EpisodesStats:
        return self._stats.stats()


class RtgAdvantage(NamedTuple):
    rewards_to_go: torch.Tensor
    advantage: torch.Tensor


class TrainEpochStats(NamedTuple):
    pi_loss: float
    v_loss: float

    def write_to_tensorboard(self, tb_writer: SummaryWriter, global_step: int) -> None:
        tb_writer.add_scalars("losses", self._asdict(), global_step=global_step)


class VanillaPolicyGradient(Algorithm):
    def __init__(
        self,
        policy: VPGActorCritic,
        env: VecEnv,
        device: torch.device,
        tb_writer: SummaryWriter,
        gamma: float = 0.99,
        pi_lr: float = 3e-4,
        val_lr: float = 1e-3,
        train_v_iters: int = 80,
        lam: float = 0.97,
        max_grad_norm: float = 10.0,
        steps_per_epoch: int = 4_000,
    ) -> None:
        super().__init__(policy, env, device, tb_writer)
        self.policy = policy

        self.gamma = gamma
        self.lam = lam
        self.pi_optim = Adam(self.policy.pi.parameters(), lr=pi_lr)
        self.val_optim = Adam(self.policy.v.parameters(), lr=val_lr)
        self.max_grad_norm = max_grad_norm

        self.steps_per_epoch = steps_per_epoch
        self.train_v_iters = train_v_iters

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Callback] = None,
    ) -> List[EpisodesStats]:
        self.policy.train(True)
        obs = self.env.reset()
        timesteps_elapsed = 0
        episodes_stats: List[EpisodesStats] = []
        while timesteps_elapsed < total_timesteps:
            accumulator = self._collect_trajectories(obs)
            epoch_stats = self.train(accumulator.trajectories)
            epoch_steps = accumulator.n_timesteps()
            timesteps_elapsed += epoch_steps
            stats = accumulator.stats()
            episodes_stats.append(stats)
            stats.write_to_tensorboard(
                self.tb_writer, "train", global_step=timesteps_elapsed
            )
            epoch_stats.write_to_tensorboard(
                self.tb_writer, global_step=timesteps_elapsed
            )
            print(
                f"Epoch: {len(episodes_stats)} | "
                f"Score: {stats.score} | "
                f"Length: {stats.length} | "
                f"Pi Loss: {round(epoch_stats.pi_loss, 2)} | "
                f"V Loss: {round(epoch_stats.v_loss, 2)} | "
                f"Total Steps: {timesteps_elapsed}"
            )
            if callback:
                callback.on_step(timesteps_elapsed=epoch_steps)
        return episodes_stats

    def train(self, trajectories: Sequence[Trajectory]) -> TrainEpochStats:
        obs = torch.as_tensor(
            np.concatenate([np.array(t.obs) for t in trajectories]), device=self.device
        )
        act = torch.as_tensor(
            np.concatenate([np.array(t.act) for t in trajectories]), device=self.device
        )
        rtg, adv = self._compute_rtg_and_advantage(trajectories)

        pi_loss = self._update_pi(obs, act, adv)
        v_loss = 0
        for _ in range(self.train_v_iters):
            v_loss = self._update_v(obs, rtg)

        return TrainEpochStats(pi_loss, v_loss)

    def _collect_trajectories(self, obs: VecEnvObs) -> TrajectoryAccumulator:
        accumulator = TrajectoryAccumulator(self.env.num_envs, self.steps_per_epoch)
        while not accumulator.is_done():
            action, value, _ = self.policy.step(obs)
            next_obs, reward, done, _ = self.env.step(action)
            accumulator.step(obs, action, reward, done, value)
            obs = next_obs
        return accumulator

    def _compute_rtg_and_advantage(
        self, trajectories: Sequence[Trajectory]
    ) -> RtgAdvantage:
        rewards_to_go = []
        advantage = []
        for traj in trajectories:
            last_val = 0 if traj.terminated else self.policy.step(traj.obs[-1]).v
            rew = np.append(np.array(traj.rew), last_val)
            v = np.append(np.array(traj.v), last_val)
            rewards_to_go.append(discounted_cumsum(rew, self.gamma)[:-1])
            deltas = rew[:-1] + self.gamma * v[1:] - v[:-1]
            advantage.append(discounted_cumsum(deltas, self.gamma * self.lam))
        return RtgAdvantage(
            torch.as_tensor(
                np.concatenate(rewards_to_go), dtype=torch.float32, device=self.device
            ),
            torch.as_tensor(
                np.concatenate(advantage), dtype=torch.float32, device=self.device
            ),
        )

    def _update_pi(
        self, obs: torch.Tensor, act: torch.Tensor, adv: torch.Tensor
    ) -> float:
        self.pi_optim.zero_grad()
        _, logp, _ = self.policy.pi(obs, act)
        pi_loss = -(logp * adv).mean()
        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.pi.parameters(), self.max_grad_norm)
        self.pi_optim.step()
        return pi_loss.item()

    def _update_v(self, obs: torch.Tensor, rtg: torch.Tensor) -> float:
        self.val_optim.zero_grad()
        v = self.policy.v(obs)
        v_loss = ((v - rtg) ** 2).mean()
        v_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.v.parameters(), self.max_grad_norm)
        self.val_optim.step()
        return v_loss.item()
