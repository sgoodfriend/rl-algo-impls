from dataclasses import dataclass
from typing import Dict, List, Optional, TypeVar, Union

import numpy as np

from rl_algo_impls.shared.gae import compute_advantages


@dataclass
class Trajectory:
    obs: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    logprobs: np.ndarray
    actions: Union[np.ndarray, Dict[str, np.ndarray]]
    action_masks: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]

    def __len__(self) -> int:
        return len(self.obs)


class TrajectoryBuilder:
    def __init__(self) -> None:
        self.reset()

    def __len__(self) -> int:
        return len(self.obs)

    def add(
        self,
        obs: np.ndarray,
        reward: Union[float, np.ndarray],
        done: bool,
        value: Union[float, np.ndarray],
        logprob: float,
        action: Union[np.ndarray, Dict[str, np.ndarray]],
        action_mask: Optional[Union[np.ndarray, Dict[str, np.ndarray]]],
    ) -> None:
        self.obs.append(obs)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.logprobs.append(logprob)
        self.actions.append(action)
        self.action_masks.append(action_mask)

    def reset(self) -> None:
        self.obs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.logprobs = []
        self.actions = []
        self.action_masks = []

    def trajectory(
        self,
        gamma: Union[float, np.ndarray],
        gae_lambda: Union[float, np.ndarray],
        next_values: Optional[np.ndarray] = None,
    ) -> Trajectory:
        np_obs = np.array(self.obs)
        np_rewards = np.array(self.rewards, dtype=np.float32)
        np_dones = np.array(self.dones)
        np_values = np.array(self.values)
        np_logprobs = np.array(self.logprobs)
        np_actions = batch_actions(self.actions)
        np_action_masks = batch_actions(self.action_masks)

        # The first element is unused so it can be anything. True is not technically
        # correct because the beginning of a trajectory doesn't necessarily mean the
        # start of an episode given that rollouts can be split across episodes.
        episode_starts = np.concatenate([[True], np_dones[:-1]])

        return Trajectory(
            obs=np_obs,
            values=np_values,
            advantages=compute_advantages(
                np_rewards,
                np_values,
                episode_starts,
                np.array(np_dones[-1]),
                next_values
                if next_values is not None
                else np.zeros(np_values.shape[1:], dtype=np_values.dtype),
                gamma,
                gae_lambda,
            ),
            logprobs=np_logprobs,
            actions=np_actions,
            action_masks=np_action_masks,
        )


ND = TypeVar("ND", np.ndarray, Dict[str, np.ndarray], None)


def batch_actions(actions: List[ND]) -> ND:
    if isinstance(actions[0], dict):
        return {k: np.array([a[k] for a in actions]) for k in actions[0]}
    elif actions[0] is None:
        return None
    return np.array(actions)
