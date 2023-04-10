import copy
import random
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.wrappers.action_mask_wrapper import find_action_masker
from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnvObs,
    VecEnvStepReturn,
    VecotarableWrapper,
)


class SelfPlayWrapper(VecotarableWrapper):
    next_obs: VecEnvObs
    next_action_masks: Optional[np.ndarray]

    def __init__(
        self,
        env,
        num_old_policies: int = 0,
        save_steps: int = 20_000,
        swap_steps: int = 10_000,
        window: int = 10,
        swap_window_size: int = 2,
    ) -> None:
        super().__init__(env)
        assert num_old_policies % 2 == 0, f"num_old_policies must be even"
        assert (
            num_old_policies % swap_window_size == 0
        ), f"num_old_policies must be a multiple of swap_window_size"

        self.num_old_policies = num_old_policies
        self.save_steps = save_steps
        self.swap_steps = swap_steps
        self.swap_window_size = swap_window_size

        self.policies: Deque[Policy] = deque(maxlen=window)
        self.policy_assignments: List[Optional[Policy]] = [None] * env.num_envs
        self.steps_since_swap = np.zeros(env.num_envs)

        self.num_envs = env.num_envs - num_old_policies

    def get_action_mask(self) -> Optional[np.ndarray]:
        return self.env.get_action_mask()[self.learner_indexes()]

    def learner_indexes(self) -> List[int]:
        return [p is None for p in self.policy_assignments]

    def checkpoint_policy(self, copied_policy: Policy) -> None:
        copied_policy.train(False)
        self.policies.append(copied_policy)

        if all(p is None for p in self.policy_assignments):
            for i in range(self.num_old_policies):
                # Switch between player 1 and 2
                self.policy_assignments[2 * i + (i % 2)] = copied_policy

    def swap_policy(self, idx: int, swap_window_size: int = 1) -> None:
        policy = random.choice(self.policies)
        idx = idx // 2 * 2
        for j in range(swap_window_size * 2):
            if self.policy_assignments[idx + j]:
                self.policy_assignments[idx + j] = policy
        self.steps_since_swap[idx : idx + swap_window_size * 2] = np.zeros(
            swap_window_size * 2
        )

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        env = self.env  # type: ignore
        all_actions = np.zeros((env.num_envs,) + actions.shape[1:], dtype=actions.dtype)
        orig_learner_indexes = self.learner_indexes()

        all_actions[orig_learner_indexes] = actions
        for policy in set(p for p in self.policy_assignments if p):
            policy_indexes = [policy == p for p in self.policy_assignments]
            if any(policy_indexes):
                all_actions[policy_indexes] = policy.act(
                    self.next_obs[policy_indexes],
                    deterministic=False,
                    action_masks=self.next_action_masks[policy_indexes]
                    if self.next_action_masks is not None
                    else None,
                )
        self.next_obs, rew, done, info = env.step(all_actions)
        self.next_action_masks = self.env.get_action_mask()

        rew = rew[orig_learner_indexes]
        info = [i for i, b in zip(info, orig_learner_indexes) if b]

        self.steps_since_swap += 1
        for idx in range(0, self.num_old_policies * 2, 2 * self.swap_window_size):
            if self.steps_since_swap[idx] > self.swap_steps:
                self.swap_policy(idx, self.swap_window_size)

        new_learner_indexes = self.learner_indexes()
        return self.next_obs[new_learner_indexes], rew, done[new_learner_indexes], info

    def reset(self) -> VecEnvObs:
        self.next_obs = super().reset()
        self.next_action_masks = self.env.get_action_mask()
        return self.next_obs[self.learner_indexes()]