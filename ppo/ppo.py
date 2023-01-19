import numpy as np
import torch
import torch.nn as nn

from dataclasses import asdict, dataclass
from torch.optim import Adam
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs
from torch.utils.tensorboard.writer import SummaryWriter
from typing import List, Optional, Sequence, NamedTuple

from shared.algorithm import Algorithm
from shared.callbacks.callback import Callback
from shared.policy.on_policy import ActorCritic
from shared.schedule import constant_schedule, linear_schedule
from shared.stats import RolloutStats, EpisodesStats
from shared.trajectory import Trajectory as BaseTrajectory
from shared.utils import discounted_cumsum


@dataclass
class PPOTrajectory(BaseTrajectory):
    logp_a: List[float]
    next_obs: Optional[np.ndarray]

    def __init__(self) -> None:
        super().__init__()
        self.logp_a = []
        self.next_obs = None

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
        super().add(obs, act, rew, v)
        self.next_obs = next_obs if not terminated else None
        self.terminated = terminated
        self.logp_a.append(logp_a)


class TrajectoryAccumulator:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs

        self.trajectories_ = []
        self.current_trajectories_ = [PPOTrajectory() for _ in range(num_envs)]

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
        assert isinstance(obs, np.ndarray)
        assert isinstance(next_obs, np.ndarray)
        for i, trajectory in enumerate(self.current_trajectories_):
            # TODO: Eventually take advantage of terminated/truncated differentiation in
            # later versions of gym.
            trajectory.add(
                obs[i], action[i], next_obs[i], reward[i], done[i], val[i], logp_a[i]
            )
            if done[i]:
                self.trajectories_.append(trajectory)
                self.current_trajectories_[i] = PPOTrajectory()

    @property
    def all_trajectories(self) -> List[PPOTrajectory]:
        return self.trajectories_ + list(
            filter(lambda t: len(t), self.current_trajectories_)
        )


class RtgAdvantage(NamedTuple):
    rewards_to_go: torch.Tensor
    advantage: torch.Tensor


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
        print_n_episodes: int = 100,
        update_rtg_between_epochs: bool = False,
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

        self.rollout_stats = RolloutStats(
            self.env.num_envs, print_n_episodes, tb_writer
        )
        self.update_rtg_between_epochs = update_rtg_between_epochs

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Callback] = None,
    ) -> List[EpisodesStats]:
        self.policy.train(True)
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

        return self.rollout_stats.epochs

    def _collect_trajectories(self, obs: VecEnvObs) -> TrajectoryAccumulator:
        accumulator = TrajectoryAccumulator(self.env.num_envs)
        for _ in range(self.n_steps):
            action, value, logp_a = self.policy.step(obs)
            next_obs, reward, done, _ = self.env.step(action)
            accumulator.step(obs, action, next_obs, reward, done, value, logp_a)
            unnormalized_reward = (
                self.policy.vec_normalize.unnormalize_reward(reward)
                if self.policy.vec_normalize
                else reward
            )
            self.rollout_stats.step(unnormalized_reward, done)
            obs = next_obs
        return accumulator

    def train(self, trajectories: List[PPOTrajectory], progress: float) -> TrainStats:
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
        if self.update_rtg_between_epochs:
            rtg = None
        else:
            rtg = self._compute_rtg(trajectories)
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
                rtg, adv = self._compute_rtg_and_advantage(trajectories)
            else:
                adv = self._compute_advantage(trajectories)
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
                        mb_adv,
                        rtg[mb_idxs],  # type: ignore
                        adv[mb_idxs],
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
        pi, logp = self.policy.pi(obs, act)
        logratio = logp - orig_logp_a
        ratio = torch.exp(logratio)
        clip_ratio = torch.clamp(ratio, min=1 - pi_clip, max=1 + pi_clip)
        pi_loss = torch.maximum(-ratio * adv, -clip_ratio * adv).mean()

        v = self.policy.v(obs)
        v_loss = (v - rtg).pow(2)
        if v_clip:
            v_clipped = (torch.clamp(v, orig_v - v_clip, orig_v + v_clip) - rtg).pow(2)
            v_loss = torch.maximum(v_loss, v_clipped)
        v_loss = v_loss.mean()

        entropy_loss = pi.entropy().mean()

        loss = pi_loss - ent_coef * entropy_loss + self.vf_coef * v_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean().cpu().numpy().item()
            clipped_frac = int(
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

    def _compute_advantage(self, trajectories: Sequence[PPOTrajectory]) -> torch.Tensor:
        advantage = []
        for traj in trajectories:
            last_val = 0
            if not traj.terminated and traj.next_obs is not None:
                with torch.no_grad():
                    next_obs = torch.as_tensor(np.array(traj.next_obs)).to(self.device)
                    last_val = self.policy.v(next_obs).cpu().numpy()
            rew = np.append(np.array(traj.rew), last_val)
            v = np.append(np.array(traj.v), last_val)
            deltas = rew[:-1] + self.gamma * v[1:] - v[:-1]
            advantage.append(discounted_cumsum(deltas, self.gamma * self.gae_lambda))
        return torch.as_tensor(
            np.concatenate(advantage), dtype=torch.float32, device=self.device
        )

    def _compute_rtg(self, trajectories: Sequence[PPOTrajectory]) -> torch.Tensor:
        rewards_to_go = []
        for traj in trajectories:
            last_val = 0
            if not traj.terminated and traj.next_obs is not None:
                with torch.no_grad():
                    next_obs = torch.as_tensor(np.array(traj.next_obs)).to(self.device)
                    last_val = self.policy.v(next_obs).cpu().numpy()
            rew = np.append(np.array(traj.rew), last_val)
            rewards_to_go.append(discounted_cumsum(rew, self.gamma)[:-1])
        return torch.as_tensor(
            np.concatenate(rewards_to_go), dtype=torch.float32, device=self.device
        )

    def _compute_rtg_and_advantage(
        self, trajectories: Sequence[PPOTrajectory]
    ) -> RtgAdvantage:
        rewards_to_go = []
        advantage = []
        for traj in trajectories:
            last_val = 0
            if not traj.terminated and traj.next_obs is not None:
                with torch.no_grad():
                    next_obs = torch.as_tensor(np.array(traj.next_obs)).to(self.device)
                    last_val = self.policy.v(next_obs).cpu().numpy()
            rew = np.append(np.array(traj.rew), last_val)
            rewards_to_go.append(discounted_cumsum(rew, self.gamma)[:-1])
            v = np.append(np.array(traj.v), last_val)
            deltas = rew[:-1] + self.gamma * v[1:] - v[:-1]
            advantage.append(discounted_cumsum(deltas, self.gamma * self.gae_lambda))
        return RtgAdvantage(
            torch.as_tensor(
                np.concatenate(rewards_to_go), dtype=torch.float32, device=self.device
            ),
            torch.as_tensor(
                np.concatenate(advantage), dtype=torch.float32, device=self.device
            ),
        )
