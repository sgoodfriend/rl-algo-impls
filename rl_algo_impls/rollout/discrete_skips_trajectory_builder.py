from typing import Dict, Optional, Union

import numpy as np

from rl_algo_impls.rollout.trajectory import Trajectory, batch_actions
from rl_algo_impls.shared.tensor_utils import prepend_dims_to_match


class DiscreteSkipsTrajectoryBuilder:
    def __init__(self) -> None:
        self.reset()

    def __len__(self) -> int:
        return len(self.obs)

    def reset(self) -> None:
        self.obs = []
        self.rewards = []
        self.done = False
        self.values = []
        self.logprobs = []
        self.actions = []
        self.action_masks = []
        self.steps_elapsed = []

    def step_no_add(
        self,
        reward: Union[float, np.ndarray],
        done: bool,
        gamma: Union[float, np.ndarray],
    ) -> None:
        assert not self.done, f"Shouldn't be stepping a done trajectory"
        if self.rewards:
            self.rewards[-1] += reward * gamma ** self.steps_elapsed[-1]
        if self.steps_elapsed:
            self.steps_elapsed[-1] += 1
        self.done = done

    def step_add(
        self,
        obs: np.ndarray,
        reward: Union[float, np.ndarray],
        done: bool,
        value: Union[float, np.ndarray],
        logprob: float,
        action: Union[np.ndarray, Dict[str, np.ndarray]],
        action_mask: Optional[Union[np.ndarray, Dict[str, np.ndarray]]],
        gamma: Union[float, np.ndarray],
    ) -> None:
        assert not self.done, f"Shouldn't be adding to a done trajectory"

        self.obs.append(obs)
        self.values.append(value)
        self.logprobs.append(logprob)
        self.actions.append(action)
        self.action_masks.append(action_mask)

        self.rewards.append(np.zeros_like(reward))
        self.steps_elapsed.append(0)
        self.step_no_add(reward, done, gamma)

    def trajectory(
        self,
        gamma: Union[float, np.ndarray],
        gae_lambda: Union[float, np.ndarray],
        next_values: Optional[np.ndarray] = None,
    ) -> Trajectory:
        assert (
            self.done or next_values is not None
        ), f"Need next_values if trajectory isn't done"

        obs = np.array(self.obs)
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values)
        logprobs = np.array(self.logprobs)
        actions = batch_actions(self.actions)
        action_masks = batch_actions(self.action_masks)
        steps_elapsed = np.array(self.steps_elapsed)

        advantages = np.zeros_like(rewards)
        last_advantage = np.zeros_like(advantages[-1])
        n_steps = advantages.shape[0]
        if isinstance(gamma, np.ndarray):
            gamma = prepend_dims_to_match(gamma, values.shape[1:])
        if isinstance(gae_lambda, np.ndarray):
            gae_lambda = prepend_dims_to_match(gae_lambda, values.shape[1:])
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = np.zeros_like(values[t]) if self.done else next_values
            else:
                next_value = values[t + 1]
            delta = rewards[t] + gamma ** steps_elapsed[t] * next_value - values[t]
            last_advantage = (
                delta + gamma ** steps_elapsed[t] * gae_lambda * last_advantage
            )
            advantages[t] = last_advantage

        return Trajectory(
            obs=obs,
            values=values,
            advantages=advantages,
            logprobs=logprobs,
            actions=actions,
            action_masks=action_masks,
        )
