import numpy as np
import torch
import torch.nn as nn

from dataclasses import asdict, dataclass, field
from torch.optim import Adam
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs
from torch.utils.tensorboard.writer import SummaryWriter
from typing import List, Optional, Sequence, NamedTuple, TypeVar

from shared.algorithm import Algorithm
from shared.callbacks.callback import Callback
from shared.gae import compute_advantage, compute_rtg_and_advantage, RtgAdvantage
from shared.policy.on_policy import ActorCritic
from shared.schedule import constant_schedule, linear_schedule
from shared.trajectory import Trajectory, TrajectoryAccumulator


@dataclass
class PPOTrajectory(Trajectory):
    logp_a: List[float] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        next_obs: np.ndarray,
        rew: float,
        terminated: bool,
        v: float,
        logp_a: float,
    ):
        super().add(obs, act, next_obs, rew, terminated, v)
        self.logp_a.append(logp_a)


class PPOTrajectoryAccumulator(TrajectoryAccumulator):
    def __init__(self, num_envs: int) -> None:
        super().__init__(num_envs, PPOTrajectory)

    def step(
        self,
        obs: VecEnvObs,
        action: np.ndarray,
        next_obs: VecEnvObs,
        reward: np.ndarray,
        done: np.ndarray,
        val: np.ndarray,
        logp_a: np.ndarray,
    ) -> None:
        super().step(obs, action, next_obs, reward, done, val, logp_a)


class TrainStepStats(NamedTuple):
    loss: float
    pi_loss: float
    v_loss: float
    entropy_loss: float
    approx_kl: float
    clipped_frac: float


@dataclass
class TrainStats:
    loss: float
    pi_loss: float
    v_loss: float
    entropy_loss: float
    approx_kl: float
    clipped_frac: float

    def __init__(self, step_stats: List[TrainStepStats]) -> None:
        self.loss = np.mean([s.loss for s in step_stats]).item()
        self.pi_loss = np.mean([s.pi_loss for s in step_stats]).item()
        self.v_loss = np.mean([s.v_loss for s in step_stats]).item()
        self.entropy_loss = np.mean([s.entropy_loss for s in step_stats]).item()
        self.approx_kl = np.mean([s.approx_kl for s in step_stats]).item()
        self.clipped_frac = np.mean([s.clipped_frac for s in step_stats]).item()

    def write_to_tensorboard(self, tb_writer: SummaryWriter, global_step: int) -> None:
        tb_writer.add_scalars("losses", asdict(self), global_step=global_step)

    def __repr__(self) -> str:
        return " | ".join(
            [
                f"Loss: {round(self.loss, 2)}",
                f"Pi L: {round(self.pi_loss, 2)}",
                f"V L: {round(self.v_loss, 2)}",
                f"E L: {round(self.entropy_loss, 2)}",
                f"Apx KL Div: {round(self.approx_kl, 2)}",
                f"Clip Frac: {round(self.clipped_frac, 2)}",
            ]
        )


PPOSelf = TypeVar("PPOSelf", bound="PPO")


class PPO(Algorithm):
    def __init__(
        self,
        policy: ActorCritic,
        env: VecEnv,
        device: torch.device,
        tb_writer: SummaryWriter,
        learning_rate: float = 3e-4,
        learning_rate_decay: str = "none",
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_decay: str = "none",
        clip_range_vf: Optional[float] = None,
        clip_range_vf_decay: str = "none",
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        ent_coef_decay: str = "none",
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_rtg_between_epochs: bool = False,
        sde_sample_freq: int = -1,
    ) -> None:
        super().__init__(policy, env, device, tb_writer)
        self.policy = policy

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.optimizer = Adam(self.policy.parameters(), lr=learning_rate)
        self.lr_schedule = (
            linear_schedule(learning_rate, 0)
            if learning_rate_decay == "linear"
            else constant_schedule(learning_rate)
        )
        self.max_grad_norm = max_grad_norm
        self.clip_range_schedule = (
            linear_schedule(clip_range, 0)
            if clip_range_decay == "linear"
            else constant_schedule(clip_range)
        )
        self.clip_range_vf_schedule = None
        if clip_range_vf:
            self.clip_range_vf_schedule = (
                linear_schedule(clip_range_vf, 0)
                if clip_range_vf_decay == "linear"
                else constant_schedule(clip_range_vf)
            )
        self.normalize_advantage = normalize_advantage
        self.ent_coef_schedule = (
            linear_schedule(ent_coef, 0)
            if ent_coef_decay == "linear"
            else constant_schedule(ent_coef)
        )
        self.vf_coef = vf_coef

        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.sde_sample_freq = sde_sample_freq

        self.update_rtg_between_epochs = update_rtg_between_epochs

    def learn(
        self: PPOSelf,
        total_timesteps: int,
        callback: Optional[Callback] = None,
    ) -> PPOSelf:
        obs = self.env.reset()
        ts_elapsed = 0
        while ts_elapsed < total_timesteps:
            accumulator = self._collect_trajectories(obs)
            progress = ts_elapsed / total_timesteps
            train_stats = self.train(accumulator.all_trajectories, progress)
            rollout_steps = self.n_steps * self.env.num_envs
            ts_elapsed += rollout_steps
            train_stats.write_to_tensorboard(self.tb_writer, ts_elapsed)
            if callback:
                callback.on_step(timesteps_elapsed=rollout_steps)

        return self

    def _collect_trajectories(self, obs: VecEnvObs) -> PPOTrajectoryAccumulator:
        self.policy.eval()
        accumulator = PPOTrajectoryAccumulator(self.env.num_envs)
        self.policy.reset_noise()
        for i in range(self.n_steps):
            if self.sde_sample_freq > 0 and i > 0 and i % self.sde_sample_freq == 0:
                self.policy.reset_noise()
            action, value, logp_a, clamped_action = self.policy.step(obs)
            next_obs, reward, done, _ = self.env.step(clamped_action)
            accumulator.step(obs, action, next_obs, reward, done, value, logp_a)
            obs = next_obs
        return accumulator

    def train(self, trajectories: List[PPOTrajectory], progress: float) -> TrainStats:
        self.policy.train()
        learning_rate = self.lr_schedule(progress)
        self.optimizer.param_groups[0]["lr"] = learning_rate

        pi_clip = self.clip_range_schedule(progress)
        v_clip = (
            self.clip_range_vf_schedule(progress)
            if self.clip_range_vf_schedule
            else None
        )
        ent_coef = self.ent_coef_schedule(progress)

        obs = torch.as_tensor(
            np.concatenate([np.array(t.obs) for t in trajectories]), device=self.device
        )
        act = torch.as_tensor(
            np.concatenate([np.array(t.act) for t in trajectories]), device=self.device
        )
        rtg, adv = compute_rtg_and_advantage(
            trajectories, self.policy, self.gamma, self.gae_lambda, self.device
        )
        orig_v = torch.as_tensor(
            np.concatenate([np.array(t.v) for t in trajectories]), device=self.device
        )
        orig_logp_a = torch.as_tensor(
            np.concatenate([np.array(t.logp_a) for t in trajectories]),
            device=self.device,
        )

        step_stats = []
        for _ in range(self.n_epochs):
            if self.update_rtg_between_epochs:
                rtg, adv = compute_rtg_and_advantage(
                    trajectories, self.policy, self.gamma, self.gae_lambda, self.device
                )
            else:
                adv = compute_advantage(
                    trajectories, self.policy, self.gamma, self.gae_lambda, self.device
                )
            idxs = torch.randperm(len(obs))
            for i in range(0, len(obs), self.batch_size):
                mb_idxs = idxs[i : i + self.batch_size]
                mb_adv = adv[mb_idxs]
                if self.normalize_advantage:
                    mb_adv = (mb_adv - mb_adv.mean(-1)) / (mb_adv.std(-1) + 1e-8)
                step_stats.append(
                    self._train_step(
                        pi_clip,
                        v_clip,
                        ent_coef,
                        obs[mb_idxs],
                        act[mb_idxs],
                        rtg[mb_idxs],
                        mb_adv,
                        orig_v[mb_idxs],
                        orig_logp_a[mb_idxs],
                    )
                )

        return TrainStats(step_stats)

    def _train_step(
        self,
        pi_clip: float,
        v_clip: Optional[float],
        ent_coef: float,
        obs: torch.Tensor,
        act: torch.Tensor,
        rtg: torch.Tensor,
        adv: torch.Tensor,
        orig_v: torch.Tensor,
        orig_logp_a: torch.Tensor,
    ) -> TrainStepStats:
        logp_a, entropy, v = self.policy(obs, act)
        logratio = logp_a - orig_logp_a
        ratio = torch.exp(logratio)
        clip_ratio = torch.clamp(ratio, min=1 - pi_clip, max=1 + pi_clip)
        pi_loss = torch.maximum(-ratio * adv, -clip_ratio * adv).mean()

        v_loss = (v - rtg).pow(2)
        if v_clip:
            v_clipped = (torch.clamp(v, orig_v - v_clip, orig_v + v_clip) - rtg).pow(2)
            v_loss = torch.maximum(v_loss, v_clipped)
        v_loss = v_loss.mean()

        entropy_loss = entropy.mean()

        loss = pi_loss - ent_coef * entropy_loss + self.vf_coef * v_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean().cpu().numpy().item()
            clipped_frac = (
                ((ratio - 1).abs() > pi_clip).float().mean().cpu().numpy().item()
            )
        return TrainStepStats(
            loss.item(),
            pi_loss.item(),
            v_loss.item(),
            entropy_loss.item(),
            approx_kl,
            clipped_frac,
        )
