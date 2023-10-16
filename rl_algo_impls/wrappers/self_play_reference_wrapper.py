from abc import ABC, abstractmethod
from typing import Callable, Generic, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium.experimental.vector.utils import batch_space

from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.tensor_utils import batch_dict_keys
from rl_algo_impls.wrappers.vector_wrapper import (
    ObsType,
    VecEnvMaskedResetReturn,
    VecEnvStepReturn,
    VectorEnv,
    VectorWrapper,
    filter_info,
)


class AbstractSelfPlayReferenceWrapper(VectorWrapper, ABC, Generic[ObsType]):
    next_obs: ObsType

    def __init__(self, env: VectorEnv) -> None:
        super().__init__(env)
        assert self.env.num_envs % 2 == 0

        self._num_envs = self.env.num_envs // 2
        self._observation_space = batch_space(
            env.single_observation_space, n=self._num_envs
        )
        self._action_space = batch_space(env.single_action_space, n=self._num_envs)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        env = self.env  # type: ignore
        all_actions = np.zeros((env.num_envs,) + actions.shape[1:], dtype=actions.dtype)

        policy_assignments, learner_indexes = self._assignment_and_indexes()

        all_actions[learner_indexes] = actions
        for policy in set(p for p in policy_assignments if p):
            policy_indexes = [policy == p for p in policy_assignments]
            all_actions[policy_indexes] = policy.act(
                self.next_obs[policy_indexes],  # type: ignore
                deterministic=False,
                action_masks=batch_dict_keys(self.next_action_masks[policy_indexes])
                if self.next_action_masks is not None
                else None,
            )
        self.next_obs, rew, terminations, truncations, info = env.step(all_actions)
        self.next_action_masks = self.env.get_action_mask()  # type: ignore

        return (
            self.next_obs[learner_indexes],
            rew[learner_indexes],
            terminations[learner_indexes],
            truncations[learner_indexes],
            filter_info(info, learner_indexes),
        )

    def reset(self, **kwargs):
        self.next_obs, info = super().reset(**kwargs)
        self.next_action_masks = self.env.get_action_mask()  # type: ignore
        _, indexes = self._assignment_and_indexes()
        return self.next_obs[indexes], filter_info(info, indexes)

    def masked_reset(self, env_mask: np.ndarray) -> VecEnvMaskedResetReturn:
        _, learner_indexes = self._assignment_and_indexes()
        mapped_mask = np.zeros_like(learner_indexes)
        mapped_mask[learner_indexes] = env_mask
        assert np.all(
            mapped_mask[::2] == mapped_mask[1::2]
        ), f"Expected mapped_mask to be the same for player 1 and 2: {mapped_mask}"
        return self.env.masked_reset(mapped_mask)  # type: ignore

    def get_action_mask(self) -> Optional[np.ndarray]:
        _, indexes = self._assignment_and_indexes()
        return (
            self.next_action_masks[indexes]
            if self.next_action_masks is not None
            else None
        )

    @abstractmethod
    def _assignment_and_indexes(self) -> Tuple[List[Optional[Policy]], List[bool]]:
        ...


class SelfPlayReferenceWrapper(AbstractSelfPlayReferenceWrapper):
    policies_getter_fn: Optional[Callable[[], Sequence[Policy]]]

    def __init__(self, env: VectorEnv, window: int) -> None:
        super().__init__(env)
        self.window = window
        self.policies_getter_fn = None

    def _assignment_and_indexes(self) -> Tuple[List[Optional[Policy]], List[bool]]:
        assignments: List[Optional[Policy]] = [None] * self.env.num_envs  # type: ignore
        policies = (
            list(reversed(self.policies_getter_fn()))
            if self.policies_getter_fn
            else [None] * self.env.num_envs
        )
        for i in range(self.num_envs):
            policy = policies[i % len(policies)]
            assignments[2 * i + (i % 2)] = policy
        return assignments, [p is None for p in assignments]

    def close(self) -> None:
        self.policies_getter_fn = None
        super().close()
