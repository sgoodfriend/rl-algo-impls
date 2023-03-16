import numpy as np
import torch

from typing import NamedTuple, Sequence

from rl_algo_impls.shared.policy.on_policy import OnPolicy
from rl_algo_impls.shared.trajectory import Trajectory


class RtgAdvantage(NamedTuple):
    rewards_to_go: torch.Tensor
    advantage: torch.Tensor


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    dc = x.copy()
    for i in reversed(range(len(x) - 1)):
        dc[i] += gamma * dc[i + 1]
    return dc


def compute_advantage(
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


def compute_rtg_and_advantage(
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
