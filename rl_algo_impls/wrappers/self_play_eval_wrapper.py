import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.wrappers.action_mask_wrapper import find_action_masker
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper
from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnvObs,
    VecEnvStepReturn,
    VectorableWrapper,
)


class SelfPlayEvalWrapper(VectorableWrapper):
    train_wrapper: Optional[SelfPlayWrapper]
    next_obs: VecEnvObs
    next_action_masks: Optional[np.ndarray]

    def __init__(self, env: VectorableWrapper) -> None:
        super().__init__(env)
        assert env.num_envs % 2 == 0
        self.num_envs = env.num_envs // 2

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        assert self.train_wrapper, "Must have assigned train_wrapper"
        env = self.env  # type: ignore
        all_actions = np.zeros((env.num_envs,) + actions.shape[1:], dtype=actions.dtype)

        policy_assignments, learner_indexes = self._assignment_and_indexes()

        all_actions[learner_indexes] = actions
        for policy in set(p for p in policy_assignments if p):
            policy_indexes = [policy == p for p in policy_assignments]
            all_actions[policy_indexes] = policy.act(
                self.next_obs[policy_indexes],  # type: ignore
                deterministic=False,
                action_masks=self.next_action_masks[policy_indexes]
                if self.next_action_masks is not None
                else None,
            )
        self.next_obs, rew, done, info = env.step(all_actions)
        self.next_action_masks = env.get_action_mask()

        rew = rew[learner_indexes]
        info = [_info for _info, is_learner in zip(info, learner_indexes) if is_learner]

        return self.next_obs[learner_indexes], rew, done[learner_indexes], info

    def reset(self) -> VecEnvObs:
        self.next_obs = super().reset()
        self.next_action_masks = self.env.get_action_mask()  # type: ignore
        _, indexes = self._assignment_and_indexes()
        return self.next_obs[indexes]  # type: ignore

    def get_action_mask(self) -> Optional[np.ndarray]:
        _, indexes = self._assignment_and_indexes()
        return (
            self.next_action_masks[indexes]
            if self.next_action_masks is not None
            else None
        )

    def _assignment_and_indexes(self) -> Tuple[List[Optional[Policy]], List[bool]]:
        assignments: List[Optional[Policy]] = [None] * self.env.num_envs  # type: ignore
        policies = list(reversed(self.train_wrapper.policies))
        for i in range(self.num_envs):
            policy = policies[i % len(policies)]
            assignments[2 * i + (i % 2)] = policy
        return assignments, [p is None for p in assignments]
