import numpy as np
import torch
import torch.nn as nn

from collections import defaultdict
from dataclasses import dataclass, asdict
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Optional, Sequence, TypeVar

from shared.algorithm import Algorithm
from shared.callbacks.callback import Callback
from shared.gae import compute_rtg_and_advantage, compute_advantage
from shared.trajectory import Trajectory, TrajectoryAccumulator
from vpg.policy import VPGActorCritic


@dataclass
class TrainEpochStats:
    pi_loss: float
    v_loss: float
    envs_with_done: int = 0
    episodes_done: int = 0

    def write_to_tensorboard(self, tb_writer: SummaryWriter, global_step: int) -> None:
        tb_writer.add_scalars("losses", asdict(self), global_step=global_step)


class VPGTrajectoryAccumulator(TrajectoryAccumulator):
    def __init__(self, num_envs: int) -> None:
        super().__init__(num_envs, trajectory_class=Trajectory)
        self.completed_per_env: defaultdict[int, int] = defaultdict(int)

    def on_done(self, env_idx: int, trajectory: Trajectory) -> None:
        self.completed_per_env[env_idx] += 1


VanillaPolicyGradientSelf = TypeVar(
    "VanillaPolicyGradientSelf", bound="VanillaPolicyGradient"
)


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
        gae_lambda: float = 0.97,
        max_grad_norm: float = 10.0,
        n_steps: int = 4_000,
        sde_sample_freq: int = -1,
        update_rtg_between_v_iters: bool = False,
    ) -> None:
        super().__init__(policy, env, device, tb_writer)
        self.policy = policy

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.pi_optim = Adam(self.policy.pi.parameters(), lr=pi_lr)
        self.val_optim = Adam(self.policy.v.parameters(), lr=val_lr)
        self.max_grad_norm = max_grad_norm

        self.n_steps = n_steps
        self.train_v_iters = train_v_iters
        self.sde_sample_freq = sde_sample_freq
        self.update_rtg_between_v_iters = update_rtg_between_v_iters

    def learn(
        self: VanillaPolicyGradientSelf,
        total_timesteps: int,
        callback: Optional[Callback] = None,
    ) -> VanillaPolicyGradientSelf:
        timesteps_elapsed = 0
        epoch_cnt = 0
        while timesteps_elapsed < total_timesteps:
            epoch_cnt += 1
            accumulator = self._collect_trajectories()
            epoch_stats = self.train(accumulator.all_trajectories)
            epoch_stats.envs_with_done = len(accumulator.completed_per_env)
            epoch_stats.episodes_done = sum(accumulator.completed_per_env.values())
            epoch_steps = accumulator.n_timesteps()
            timesteps_elapsed += epoch_steps
            epoch_stats.write_to_tensorboard(
                self.tb_writer, global_step=timesteps_elapsed
            )
            print(
                f"Epoch: {epoch_cnt} | "
                f"Pi Loss: {round(epoch_stats.pi_loss, 2)} | "
                f"V Loss: {round(epoch_stats.v_loss, 2)} | "
                f"Total Steps: {timesteps_elapsed}"
            )
            if callback:
                callback.on_step(timesteps_elapsed=epoch_steps)
        return self

    def train(self, trajectories: Sequence[Trajectory]) -> TrainEpochStats:
        self.policy.train()
        obs = torch.as_tensor(
            np.concatenate([np.array(t.obs) for t in trajectories]), device=self.device
        )
        act = torch.as_tensor(
            np.concatenate([np.array(t.act) for t in trajectories]), device=self.device
        )
        rtg, adv = compute_rtg_and_advantage(
            trajectories, self.policy, self.gamma, self.gae_lambda, self.device
        )

        pi_loss = self._update_pi(obs, act, adv)
        v_loss = 0
        for _ in range(self.train_v_iters):
            if self.update_rtg_between_v_iters:
                rtg = compute_advantage(
                    trajectories, self.policy, self.gamma, self.gae_lambda, self.device
                )
            v_loss = self._update_v(obs, rtg)

        return TrainEpochStats(pi_loss, v_loss)

    def _collect_trajectories(self) -> VPGTrajectoryAccumulator:
        self.policy.eval()
        obs = self.env.reset()
        accumulator = VPGTrajectoryAccumulator(self.env.num_envs)
        self.policy.reset_noise()
        for i in range(self.n_steps):
            if self.sde_sample_freq > 0 and i > 0 and i % self.sde_sample_freq == 0:
                self.policy.reset_noise()
            action, value, _, clamped_action = self.policy.step(obs)
            next_obs, reward, done, _ = self.env.step(clamped_action)
            accumulator.step(obs, action, next_obs, reward, done, value)
            obs = next_obs
        return accumulator

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
