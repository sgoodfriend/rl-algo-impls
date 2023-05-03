import logging
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import List, NamedTuple, Optional, TypeVar

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.callbacks import Callback
from rl_algo_impls.shared.gae import compute_advantages
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.schedule import (
    constant_schedule,
    linear_schedule,
    schedule,
    update_learning_rate,
)
from rl_algo_impls.shared.stats import log_scalars
from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnv,
    single_action_space,
    single_observation_space,
)


class TrainStepStats(NamedTuple):
    loss: float
    pi_loss: float
    v_loss: float
    entropy_loss: float
    approx_kl: float
    clipped_frac: float
    val_clipped_frac: float


@dataclass
class TrainStats:
    loss: float
    pi_loss: float
    v_loss: float
    entropy_loss: float
    approx_kl: float
    clipped_frac: float
    val_clipped_frac: float
    explained_var: float

    def __init__(self, step_stats: List[TrainStepStats], explained_var: float) -> None:
        self.loss = np.mean([s.loss for s in step_stats]).item()
        self.pi_loss = np.mean([s.pi_loss for s in step_stats]).item()
        self.v_loss = np.mean([s.v_loss for s in step_stats]).item()
        self.entropy_loss = np.mean([s.entropy_loss for s in step_stats]).item()
        self.approx_kl = np.mean([s.approx_kl for s in step_stats]).item()
        self.clipped_frac = np.mean([s.clipped_frac for s in step_stats]).item()
        self.val_clipped_frac = np.mean([s.val_clipped_frac for s in step_stats]).item()
        self.explained_var = explained_var

    def write_to_tensorboard(self, tb_writer: SummaryWriter, global_step: int) -> None:
        for name, value in asdict(self).items():
            tb_writer.add_scalar(f"losses/{name}", value, global_step=global_step)

    def __repr__(self) -> str:
        return " | ".join(
            [
                f"Loss: {round(self.loss, 2)}",
                f"Pi L: {round(self.pi_loss, 2)}",
                f"V L: {round(self.v_loss, 2)}",
                f"E L: {round(self.entropy_loss, 2)}",
                f"Apx KL Div: {round(self.approx_kl, 2)}",
                f"Clip Frac: {round(self.clipped_frac, 2)}",
                f"Val Clip Frac: {round(self.val_clipped_frac, 2)}",
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
        ppo2_vf_coef_halving: bool = False,
        max_grad_norm: float = 0.5,
        sde_sample_freq: int = -1,
        update_advantage_between_epochs: bool = True,
        update_returns_between_epochs: bool = False,
        gamma_end: Optional[float] = None,
    ) -> None:
        super().__init__(policy, env, device, tb_writer)
        self.policy = policy
        self.get_action_mask = getattr(env, "get_action_mask", None)

        self.gamma_schedule = (
            linear_schedule(gamma, gamma_end)
            if gamma_end is not None
            else constant_schedule(gamma)
        )
        self.gae_lambda = gae_lambda
        self.optimizer = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-7)
        self.lr_schedule = schedule(learning_rate_decay, learning_rate)
        self.max_grad_norm = max_grad_norm
        self.clip_range_schedule = schedule(clip_range_decay, clip_range)
        self.clip_range_vf_schedule = None
        if clip_range_vf:
            self.clip_range_vf_schedule = schedule(clip_range_vf_decay, clip_range_vf)

        if normalize_advantage:
            assert (
                env.num_envs * n_steps > 1 and batch_size > 1
            ), f"Each minibatch must be larger than 1 to support normalization"
        self.normalize_advantage = normalize_advantage

        self.ent_coef_schedule = schedule(ent_coef_decay, ent_coef)
        self.vf_coef = vf_coef
        self.ppo2_vf_coef_halving = ppo2_vf_coef_halving

        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.sde_sample_freq = sde_sample_freq

        self.update_advantage_between_epochs = update_advantage_between_epochs
        self.update_returns_between_epochs = update_returns_between_epochs

    def learn(
        self: PPOSelf,
        train_timesteps: int,
        callbacks: Optional[List[Callback]] = None,
        total_timesteps: Optional[int] = None,
        start_timesteps: int = 0,
    ) -> PPOSelf:
        if total_timesteps is None:
            total_timesteps = train_timesteps
        assert start_timesteps + train_timesteps <= total_timesteps

        epoch_dim = (self.n_steps, self.env.num_envs)
        step_dim = (self.env.num_envs,)
        obs_space = single_observation_space(self.env)
        act_space = single_action_space(self.env)
        act_shape = self.policy.action_shape

        next_obs = self.env.reset()
        next_action_masks = self.get_action_mask() if self.get_action_mask else None
        next_episode_starts = np.full(step_dim, True, dtype=np.bool_)

        obs = np.zeros(epoch_dim + obs_space.shape, dtype=obs_space.dtype)  # type: ignore
        actions = np.zeros(epoch_dim + act_shape, dtype=act_space.dtype)  # type: ignore
        rewards = np.zeros(epoch_dim, dtype=np.float32)
        episode_starts = np.zeros(epoch_dim, dtype=np.bool_)
        values = np.zeros(epoch_dim, dtype=np.float32)
        logprobs = np.zeros(epoch_dim, dtype=np.float32)
        action_masks = (
            np.zeros(
                (self.n_steps,) + next_action_masks.shape, dtype=next_action_masks.dtype
            )
            if next_action_masks is not None
            else None
        )

        timesteps_elapsed = start_timesteps
        while timesteps_elapsed < start_timesteps + train_timesteps:
            start_time = perf_counter()

            progress = timesteps_elapsed / total_timesteps
            ent_coef = self.ent_coef_schedule(progress)
            learning_rate = self.lr_schedule(progress)
            update_learning_rate(self.optimizer, learning_rate)
            pi_clip = self.clip_range_schedule(progress)
            gamma = self.gamma_schedule(progress)
            chart_scalars = {
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "ent_coef": ent_coef,
                "pi_clip": pi_clip,
                "gamma": gamma,
                "gae_lambda": self.gae_lambda,
            }
            if self.clip_range_vf_schedule:
                v_clip = self.clip_range_vf_schedule(progress)
                chart_scalars["v_clip"] = v_clip
            else:
                v_clip = None
            if hasattr(self.env, "reward_weights"):
                chart_scalars["first_reward_weight"] = getattr(
                    self.env, "reward_weights"
                )[0]
            log_scalars(self.tb_writer, "charts", chart_scalars, timesteps_elapsed)

            self.policy.eval()
            self.policy.reset_noise()
            for s in range(self.n_steps):
                timesteps_elapsed += self.env.num_envs
                if self.sde_sample_freq > 0 and s > 0 and s % self.sde_sample_freq == 0:
                    self.policy.reset_noise()

                obs[s] = next_obs
                episode_starts[s] = next_episode_starts
                if action_masks is not None:
                    action_masks[s] = next_action_masks

                (
                    actions[s],
                    values[s],
                    logprobs[s],
                    clamped_action,
                ) = self.policy.step(next_obs, action_masks=next_action_masks)
                next_obs, rewards[s], next_episode_starts, _ = self.env.step(
                    clamped_action
                )
                next_action_masks = (
                    self.get_action_mask() if self.get_action_mask else None
                )

            self.policy.train()

            b_obs = torch.tensor(obs.reshape((-1,) + obs_space.shape)).to(self.device)  # type: ignore
            b_actions = torch.tensor(actions.reshape((-1,) + act_shape)).to(  # type: ignore
                self.device
            )
            b_logprobs = torch.tensor(logprobs.reshape(-1)).to(self.device)
            b_action_masks = (
                torch.tensor(action_masks.reshape((-1,) + next_action_masks.shape[1:])).to(  # type: ignore
                    self.device
                )
                if action_masks is not None
                else None
            )

            y_pred = values.reshape(-1)
            b_values = torch.tensor(y_pred).to(self.device)

            step_stats = []
            # Define variables that will definitely be set through the first epoch
            advantages: np.ndarray = None  # type: ignore
            b_advantages: torch.Tensor = None  # type: ignore
            y_true: np.ndarray = None  # type: ignore
            b_returns: torch.Tensor = None  # type: ignore
            for e in range(self.n_epochs):
                if e == 0 or self.update_advantage_between_epochs:
                    advantages = compute_advantages(
                        rewards,
                        values,
                        episode_starts,
                        next_episode_starts,
                        next_obs,
                        self.policy,
                        gamma,
                        self.gae_lambda,
                    )
                    b_advantages = torch.tensor(advantages.reshape(-1)).to(self.device)
                if e == 0 or self.update_returns_between_epochs:
                    returns = advantages + values
                    y_true = returns.reshape(-1)
                    b_returns = torch.tensor(y_true).to(self.device)

                b_idxs = torch.randperm(len(b_obs))
                # Only record last epoch's stats
                step_stats.clear()
                for i in range(0, len(b_obs), self.batch_size):
                    self.policy.reset_noise(self.batch_size)

                    mb_idxs = b_idxs[i : i + self.batch_size]

                    mb_obs = b_obs[mb_idxs]
                    mb_actions = b_actions[mb_idxs]
                    mb_values = b_values[mb_idxs]
                    mb_logprobs = b_logprobs[mb_idxs]
                    mb_action_masks = (
                        b_action_masks[mb_idxs] if b_action_masks is not None else None
                    )

                    mb_adv = b_advantages[mb_idxs]
                    if self.normalize_advantage:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    mb_returns = b_returns[mb_idxs]

                    new_logprobs, entropy, new_values = self.policy(
                        mb_obs, mb_actions, action_masks=mb_action_masks
                    )

                    logratio = new_logprobs - mb_logprobs
                    ratio = torch.exp(logratio)
                    clipped_ratio = torch.clamp(ratio, min=1 - pi_clip, max=1 + pi_clip)
                    pi_loss = torch.max(-ratio * mb_adv, -clipped_ratio * mb_adv).mean()

                    v_loss_unclipped = (new_values - mb_returns) ** 2
                    if v_clip:
                        v_loss_clipped = (
                            mb_values
                            + torch.clamp(new_values - mb_values, -v_clip, v_clip)
                            - mb_returns
                        ) ** 2
                        v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = v_loss_unclipped.mean()

                    if self.ppo2_vf_coef_halving:
                        v_loss *= 0.5

                    entropy_loss = -entropy.mean()

                    loss = pi_loss + ent_coef * entropy_loss + self.vf_coef * v_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean().cpu().numpy().item()
                        clipped_frac = (
                            ((ratio - 1).abs() > pi_clip)
                            .float()
                            .mean()
                            .cpu()
                            .numpy()
                            .item()
                        )
                        val_clipped_frac = (
                            ((new_values - mb_values).abs() > v_clip)
                            .float()
                            .mean()
                            .cpu()
                            .numpy()
                            .item()
                            if v_clip
                            else 0
                        )

                    step_stats.append(
                        TrainStepStats(
                            loss.item(),
                            pi_loss.item(),
                            v_loss.item(),
                            entropy_loss.item(),
                            approx_kl,
                            clipped_frac,
                            val_clipped_frac,
                        )
                    )

            var_y = np.var(y_true).item()
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred).item() / var_y
            )
            TrainStats(step_stats, explained_var).write_to_tensorboard(
                self.tb_writer, timesteps_elapsed
            )

            end_time = perf_counter()
            rollout_steps = self.n_steps * self.env.num_envs
            self.tb_writer.add_scalar(
                "train/steps_per_second",
                rollout_steps / (end_time - start_time),
                timesteps_elapsed,
            )

            if callbacks:
                if not all(
                    c.on_step(timesteps_elapsed=rollout_steps) for c in callbacks
                ):
                    logging.info(
                        f"Callback terminated training at {timesteps_elapsed} timesteps"
                    )
                    break

        return self
