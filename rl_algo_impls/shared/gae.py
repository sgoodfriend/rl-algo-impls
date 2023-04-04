import numpy as np
import torch

from typing import NamedTuple, Sequence

from rl_algo_impls.shared.policy.actor_critic import OnPolicy
from rl_algo_impls.shared.trajectory import Trajectory
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnvObs


class RtgAdvantage(NamedTuple):
    rewards_to_go: torch.Tensor
    advantage: torch.Tensor


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    dc = x.copy()
    for i in reversed(range(len(x) - 1)):
        dc[i] += gamma * dc[i + 1]
    return dc


def compute_advantage_from_trajectories(
    trajectories: Sequence[Trajectory],
    policy: OnPolicy,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> torch.Tensor:
    advantage = []
    for traj in trajectories:
        last_val = 0
        if not traj.terminated and traj.next_obs is not None:
            last_val = policy.value(traj.next_obs)
        rew = np.append(np.array(traj.rew), last_val)
        v = np.append(np.array(traj.v), last_val)
        deltas = rew[:-1] + gamma * v[1:] - v[:-1]
        advantage.append(discounted_cumsum(deltas, gamma * gae_lambda))
    return torch.as_tensor(
        np.concatenate(advantage), dtype=torch.float32, device=device
    )


def compute_rtg_and_advantage_from_trajectories(
    trajectories: Sequence[Trajectory],
    policy: OnPolicy,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> RtgAdvantage:
    rewards_to_go = []
    advantages = []
    for traj in trajectories:
        last_val = 0
        if not traj.terminated and traj.next_obs is not None:
            last_val = policy.value(traj.next_obs)
        rew = np.append(np.array(traj.rew), last_val)
        v = np.append(np.array(traj.v), last_val)
        deltas = rew[:-1] + gamma * v[1:] - v[:-1]
        adv = discounted_cumsum(deltas, gamma * gae_lambda)
        advantages.append(adv)
        rewards_to_go.append(v[:-1] + adv)
    return RtgAdvantage(
        torch.as_tensor(
            np.concatenate(rewards_to_go), dtype=torch.float32, device=device
        ),
        torch.as_tensor(np.concatenate(advantages), dtype=torch.float32, device=device),
    )


def compute_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_starts: np.ndarray,
    next_episode_starts: np.ndarray,
    next_obs: VecEnvObs,
    policy: OnPolicy,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0
    n_steps = advantages.shape[0]
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_nonterminal = 1.0 - next_episode_starts
            next_value = policy.value(next_obs)
        else:
            next_nonterminal = 1.0 - episode_starts[t + 1]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_nonterminal * last_gae_lam
        advantages[t] = last_gae_lam
    return advantages
