import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, TypeVar

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.callbacks import Callback
from rl_algo_impls.shared.gae import compute_rtg_and_advantage_from_trajectories
from rl_algo_impls.shared.trajectory import Trajectory, TrajectoryAccumulator
from rl_algo_impls.vpg.policy import VPGActorCritic
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv


@dataclass
class TrainEpochStats:
    pi_loss: float
    entropy_loss: float
    v_loss: float
    envs_with_done: int = 0
    episodes_done: int = 0

    def write_to_tensorboard(self, tb_writer: SummaryWriter, global_step: int) -> None:
        for name, value in asdict(self).items():
            tb_writer.add_scalar(f"losses/{name}", value, global_step=global_step)


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
        ent_coef: float = 0.0,
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

        self.ent_coef = ent_coef

    def learn(
        self: VanillaPolicyGradientSelf,
        total_timesteps: int,
        callbacks: Optional[List[Callback]] = None,
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
                " | ".join(
                    [
                        f"Epoch: {epoch_cnt}",
                        f"Pi Loss: {round(epoch_stats.pi_loss, 2)}",
                        f"Epoch Loss: {round(epoch_stats.entropy_loss, 2)}",
                        f"V Loss: {round(epoch_stats.v_loss, 2)}",
                        f"Total Steps: {timesteps_elapsed}",
                    ]
                )
            )
            if callbacks:
                if not all(c.on_step(timesteps_elapsed=epoch_steps) for c in callbacks):
                    logging.info(
                        f"Callback terminated training at {timesteps_elapsed} timesteps"
                    )
                    break
        return self

    def train(self, trajectories: Sequence[Trajectory]) -> TrainEpochStats:
        self.policy.train()
        obs = torch.as_tensor(
            np.concatenate([np.array(t.obs) for t in trajectories]), device=self.device
        )
        act = torch.as_tensor(
            np.concatenate([np.array(t.act) for t in trajectories]), device=self.device
        )
        rtg, adv = compute_rtg_and_advantage_from_trajectories(
            trajectories, self.policy, self.gamma, self.gae_lambda, self.device
        )

        _, logp, entropy = self.policy.pi(obs, act)
        pi_loss = -(logp * adv).mean()
        entropy_loss = entropy.mean()

        actor_loss = pi_loss - self.ent_coef * entropy_loss

        self.pi_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.pi.parameters(), self.max_grad_norm)
        self.pi_optim.step()

        v_loss = 0
        for _ in range(self.train_v_iters):
            v = self.policy.v(obs)
            v_loss = ((v - rtg) ** 2).mean()

            self.val_optim.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.v.parameters(), self.max_grad_norm)
            self.val_optim.step()

        return TrainEpochStats(
            pi_loss.item(),
            entropy_loss.item(),
            v_loss.item(),  # type: ignore
        )

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
