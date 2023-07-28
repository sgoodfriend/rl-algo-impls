import logging
from dataclasses import astuple
from time import perf_counter
from typing import List, Optional, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.ppo.rollout import RolloutGenerator
from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.callbacks import Callback
from rl_algo_impls.shared.gae import compute_advantages_from_policy
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.schedule import schedule, update_learning_rate
from rl_algo_impls.shared.stats import log_scalars
from rl_algo_impls.shared.tensor_utils import NumOrList, num_or_array
from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnv,
    single_action_space,
    single_observation_space,
)

A2CSelf = TypeVar("A2CSelf", bound="A2C")


class A2C(Algorithm):
    def __init__(
        self,
        policy: ActorCritic,
        device: torch.device,
        tb_writer: SummaryWriter,
        learning_rate: float = 7e-4,
        learning_rate_decay: str = "none",
        gamma: NumOrList = 0.99,
        gae_lambda: NumOrList = 1.0,
        ent_coef: float = 0.0,
        ent_coef_decay: str = "none",
        vf_coef: NumOrList = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        multi_reward_weights: Optional[List[int]] = None,
        scale_loss_by_num_actions: bool = False,
        min_logprob: Optional[float] = None,
    ) -> None:
        super().__init__(policy, device, tb_writer)
        self.policy = policy

        self.learning_rate_schedule = schedule(learning_rate_decay, learning_rate)
        if use_rms_prop:
            self.optimizer = torch.optim.RMSprop(
                policy.parameters(), lr=learning_rate, eps=rms_prop_eps
            )
        else:
            self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

        self.gamma = num_or_array(gamma)
        self.gae_lambda = num_or_array(gae_lambda)

        self.vf_coef = num_or_array(vf_coef)
        self.ent_coef_schedule = schedule(ent_coef_decay, ent_coef)
        self.max_grad_norm = max_grad_norm

        self.sde_sample_freq = sde_sample_freq
        self.normalize_advantage = normalize_advantage

        self.multi_reward_weights = (
            np.array(multi_reward_weights) if multi_reward_weights else None
        )
        self.scale_loss_by_num_actions = scale_loss_by_num_actions
        self.min_logprob = min_logprob

    def learn(
        self: A2CSelf,
        train_timesteps: int,
        rollout_generator: RolloutGenerator,
        callbacks: Optional[List[Callback]] = None,
        total_timesteps: Optional[int] = None,
        start_timesteps: int = 0,
    ) -> A2CSelf:
        if total_timesteps is None:
            total_timesteps = train_timesteps
        assert start_timesteps + train_timesteps <= total_timesteps

        timesteps_elapsed = start_timesteps
        while timesteps_elapsed < start_timesteps + train_timesteps:
            start_time = perf_counter()

            progress = timesteps_elapsed / total_timesteps
            ent_coef = self.ent_coef_schedule(progress)
            learning_rate = self.learning_rate_schedule(progress)
            update_learning_rate(self.optimizer, learning_rate)
            chart_scalars = {
                "ent_coef": ent_coef,
                "learning_rate": learning_rate,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "vf_coef": self.vf_coef,
            }

            if self.multi_reward_weights is not None:
                chart_scalars["reward_weights"] = self.multi_reward_weights
            log_scalars(self.tb_writer, "charts", chart_scalars, timesteps_elapsed)

            r = rollout_generator.rollout(self.gamma, self.gae_lambda)
            timesteps_elapsed += r.total_steps

            vf_coef = torch.Tensor(np.array(self.vf_coef)).to(self.device)

            (
                b_obs,
                _,
                b_actions,
                b_action_masks,
                b_num_actions,
                _,
                b_advantages,
                b_returns,
            ) = astuple(r.batch(self.device))

            if self.normalize_advantage:
                b_advantages = (b_advantages - b_advantages.mean()) / (
                    b_advantages.std() + 1e-8
                )

            logp_a, entropy, v = self.policy(
                b_obs, b_actions, action_masks=b_action_masks
            )

            pi_loss = -(b_advantages * logp_a)
            if self.scale_loss_by_num_actions:
                pi_loss = torch.where(b_num_actions > 0, pi_loss / b_num_actions, 0)
            if self.min_logprob is not None:
                clipped_pi_loss = -(
                    b_advantages
                    * torch.max(
                        logp_a, torch.Tensor((self.min_logprob,)).to(self.device)
                    )
                )
                pi_loss = torch.max(pi_loss, clipped_pi_loss)
            pi_loss = pi_loss.mean()

            value_loss = ((v - b_returns) ** 2).mean(0)
            entropy_loss = -entropy.mean()

            loss = pi_loss + (vf_coef * value_loss).sum() + ent_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            var_y = np.var(r.y_true).item()
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(r.y_true - r.y_pred).item() / var_y
            )

            end_time = perf_counter()
            rollout_steps = r.total_steps
            self.tb_writer.add_scalar(
                "train/steps_per_second",
                rollout_steps / (end_time - start_time),
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

            if callbacks:
                if not all(
                    c.on_step(timesteps_elapsed=rollout_steps) for c in callbacks
                ):
                    logging.info(
                        f"Callback terminated training at {timesteps_elapsed} timesteps"
                    )
                    break

        return self
