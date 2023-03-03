import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import asdict, dataclass, field
from time import perf_counter
from torch.utils.tensorboard.writer import SummaryWriter
from typing import List, Optional, Sequence, NamedTuple, TypeVar

from shared.algorithm import Algorithm
from shared.callbacks.callback import Callback
from shared.gae import compute_advantage, compute_rtg_and_advantage
from shared.policy.on_policy import ActorCritic
from shared.schedule import schedule, update_learning_rate
from shared.stats import log_scalars
from shared.trajectory import Trajectory, TrajectoryAccumulator
from wrappers.vectorable_wrapper import (
    VecEnv,
    VecEnvObs,
    single_observation_space,
    single_action_space,
)

A2CSelf = TypeVar("A2CSelf", bound="A2C")


class A2C(Algorithm):
    def __init__(
        self,
        policy: ActorCritic,
        env: VecEnv,
        device: torch.device,
        tb_writer: SummaryWriter,
        learning_rate: float = 7e-4,
        learning_rate_decay: str = "none",
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        ent_coef_decay: str = "none",
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
    ) -> None:
        super().__init__(policy, env, device, tb_writer)
        self.policy = policy

        self.lr_schedule = schedule(learning_rate_decay, learning_rate)
        if use_rms_prop:
            self.optimizer = torch.optim.RMSprop(
                policy.parameters(), lr=learning_rate, eps=rms_prop_eps
            )
        else:
            self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

        self.n_steps = n_steps

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.vf_coef = vf_coef
        self.ent_coef_schedule = schedule(ent_coef_decay, ent_coef)
        self.max_grad_norm = max_grad_norm

        self.sde_sample_freq = sde_sample_freq
        self.normalize_advantage = normalize_advantage

    def learn(
        self: A2CSelf, total_timesteps: int, callback: Optional[Callback] = None
    ) -> A2CSelf:
        epoch_dim = (self.n_steps, self.env.num_envs)
        step_dim = (self.env.num_envs,)
        obs_space = single_observation_space(self.env)
        act_space = single_action_space(self.env)

        obs = np.zeros(epoch_dim + obs_space.shape, dtype=obs_space.dtype)
        actions = np.zeros(epoch_dim + act_space.shape, dtype=act_space.dtype)
        rewards = np.zeros(epoch_dim, dtype=np.float32)
        episode_starts = np.zeros(epoch_dim, dtype=np.byte)
        values = np.zeros(epoch_dim, dtype=np.float32)
        logprobs = np.zeros(epoch_dim, dtype=np.float32)

        next_obs = self.env.reset()
        next_episode_starts = np.ones(step_dim, dtype=np.byte)

        timesteps_elapsed = 0
        while timesteps_elapsed < total_timesteps:
            start_time = perf_counter()

            progress = timesteps_elapsed / total_timesteps
            ent_coef = self.ent_coef_schedule(progress)
            learning_rate = self.lr_schedule(progress)
            update_learning_rate(self.optimizer, learning_rate)
            log_scalars(
                self.tb_writer,
                "charts",
                {
                    "ent_coef": ent_coef,
                    "learning_rate": learning_rate,
                },
                timesteps_elapsed,
            )

            self.policy.eval()
            self.policy.reset_noise()
            for s in range(self.n_steps):
                timesteps_elapsed += self.env.num_envs
                if self.sde_sample_freq > 0 and s > 0 and s % self.sde_sample_freq == 0:
                    self.policy.reset_noise()

                obs[s] = next_obs
                episode_starts[s] = next_episode_starts

                actions[s], values[s], logprobs[s], clamped_action = self.policy.step(
                    next_obs
                )
                next_obs, rewards[s], next_episode_starts, _ = self.env.step(
                    clamped_action
                )

            advantages = np.zeros(epoch_dim, dtype=np.float32)
            last_gae_lam = 0
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    next_nonterminal = 1.0 - next_episode_starts
                    next_value = self.policy.value(next_obs)
                else:
                    next_nonterminal = 1.0 - episode_starts[t + 1]
                    next_value = values[t + 1]
                delta = (
                    rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
                )
                last_gae_lam = (
                    delta
                    + self.gamma * self.gae_lambda * next_nonterminal * last_gae_lam
                )
                advantages[t] = last_gae_lam
            returns = advantages + values

            b_obs = torch.tensor(obs.reshape((-1,) + obs_space.shape)).to(self.device)
            b_actions = torch.tensor(actions.reshape((-1,) + act_space.shape)).to(
                self.device
            )
            b_advantages = torch.tensor(advantages.reshape(-1)).to(self.device)
            b_returns = torch.tensor(returns.reshape(-1)).to(self.device)

            if self.normalize_advantage:
                b_advantages = (b_advantages - b_advantages.mean()) / (
                    b_advantages.std() + 1e-8
                )

            self.policy.train()
            logp_a, entropy, v = self.policy(b_obs, b_actions)

            pi_loss = -(b_advantages * logp_a).mean()
            value_loss = F.mse_loss(b_returns, v)
            entropy_loss = -entropy.mean()

            loss = pi_loss + self.vf_coef * value_loss + ent_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            y_pred = values.reshape(-1)
            y_true = returns.reshape(-1)
            var_y = np.var(y_true).item()
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred).item() / var_y
            )

            end_time = perf_counter()
            rollout_steps = self.n_steps * self.env.num_envs
            self.tb_writer.add_scalar(
                "train/steps_per_second",
                (rollout_steps) / (end_time - start_time),
                timesteps_elapsed,
            )

            log_scalars(
                self.tb_writer,
                "losses",
                {
                    "loss": loss.item(),
                    "pi_loss": pi_loss.item(),
                    "v_loss": value_loss.item(),
                    "entropy_loss": entropy_loss.item(),
                    "explained_var": explained_var,
                },
                timesteps_elapsed,
            )

            if callback:
                if not callback.on_step(timesteps_elapsed=rollout_steps):
                    logging.info(
                        f"Callback terminated training at {timesteps_elapsed} timesteps"
                    )
                    break

        return self
