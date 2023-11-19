import collections.abc
from abc import ABC
from typing import Any, Callable, Generic, List, Optional, Sequence

import numpy as np
from gymnasium.experimental.vector.utils import batch_space

from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.tensor_utils import batch_dict_keys
from rl_algo_impls.wrappers.vector_wrapper import (
    ObsType,
    VecEnvMaskedResetReturn,
    VecEnvResetReturn,
    VecEnvStepReturn,
    VectorEnv,
    VectorWrapper,
    filter_info,
)


class PlayCheckpointsWrapper(VectorWrapper, ABC, Generic[ObsType]):
    next_obs: ObsType
    checkpoints_getter: Optional[Callable[[], Sequence[Policy]]] = None

    def __init__(
        self,
        env: VectorEnv,
        n_envs_against_checkpoints: Optional[int],
    ) -> None:
        super().__init__(env)
        if n_envs_against_checkpoints is None:
            n_envs_against_checkpoints = env.num_envs // 2
        assert n_envs_against_checkpoints <= env.num_envs // 2, (
            f"n_envs_against_checkpoints {n_envs_against_checkpoints} must be less "
            f"than or equal to half the number of envs {env.num_envs}",
        )

        self.n_envs_against_checkpoints = n_envs_against_checkpoints
        self.observation_space = batch_space(
            env.single_observation_space, n=self.num_envs
        )
        self.action_space = batch_space(env.single_action_space, n=self.num_envs)

    @property
    def num_envs(self) -> int:
        return self.env.num_envs - self.n_envs_against_checkpoints

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        all_actions = np.zeros(
            (self.env.num_envs,) + actions.shape[1:], dtype=actions.dtype
        )
        policy_assignments = self._policy_assignments()
        learning_mask = _learning_mask(policy_assignments)

        all_actions[learning_mask] = actions
        for policy in set(p for p in policy_assignments if p):
            policy_mask = [policy == _p for _p in policy_assignments]
            all_actions[policy_mask] = policy.act(
                self.next_obs[policy_mask],  # type: ignore
                deterministic=False,
                action_masks=batch_dict_keys(self.next_action_masks[policy_mask])
                if self.next_action_masks is not None
                else None,
            )
        self.next_obs, rew, terminations, truncations, info = self.env.step(all_actions)
        self.next_action_masks = self.env.get_action_mask()  # type: ignore

        return (
            self.next_obs[learning_mask],
            rew[learning_mask],
            terminations[learning_mask],
            truncations[learning_mask],
            filter_info(info, learning_mask),
        )

    def reset(self, **kwargs) -> VecEnvResetReturn:
        self.next_obs, info = super().reset(**kwargs)
        self.next_action_masks = self.env.get_action_mask()  # type: ignore
        learning_mask = _learning_mask(self._policy_assignments())
        return self.next_obs[learning_mask], filter_info(info, learning_mask)

    def masked_reset(self, env_mask: np.ndarray) -> VecEnvMaskedResetReturn:
        assert len(env_mask) == self.num_envs
        policy_assignments = self._policy_assignments()
        learning_mask = _learning_mask(policy_assignments)
        mapped_mask = np.zeros_like(learning_mask)
        mapped_mask[learning_mask] = env_mask
        # bool[N/2]
        paired_envs_mask = mapped_mask[::2] | mapped_mask[1::2]
        # bool[N]
        filled_mapped_mask = np.repeat(paired_envs_mask, 2)
        # bool[sum(paired_envs_mask)]
        env_masked_reset_mask = mapped_mask[filled_mapped_mask]
        # bool[sum(paired_envs_mask)]
        policy_assigned_mask = np.array([bool(p) for p in policy_assignments])[
            filled_mapped_mask
        ]
        assert np.logical_or(env_masked_reset_mask, policy_assigned_mask).all()
        _assert_filled_mapped_mask(filled_mapped_mask, mapped_mask, policy_assignments)
        obs, action_mask, info = self.env.masked_reset(filled_mapped_mask)  # type: ignore
        return (
            obs[env_masked_reset_mask],
            action_mask[env_masked_reset_mask],
            filter_info(info, env_masked_reset_mask),
        )

    def get_action_mask(self) -> Optional[np.ndarray]:
        if self.next_action_masks is None:
            return None
        return self.next_action_masks[_learning_mask(self._policy_assignments())]

    def __getattr__(self, name: str) -> Any:
        attr = super().__getattr__(name)
        if (
            isinstance(attr, collections.abc.Sequence)
            and not isinstance(attr, str)
            and len(attr) == self.env.num_envs
        ):
            learning_mask = _learning_mask(self._policy_assignments())
            if isinstance(attr, np.ndarray):
                return attr[learning_mask]
            else:
                return [a for a, m in zip(attr, learning_mask) if m]
        return attr

    def _policy_assignments(self) -> List[Optional[Policy]]:
        assignments: List[Optional[Policy]] = [None] * self.env.num_envs
        policies = self.checkpoints_getter() if self.checkpoints_getter else [None]
        for i in range(self.n_envs_against_checkpoints):
            # Play each policy in a pair of player 1/2 matches.
            policy = policies[(i // 2) % len(policies)]
            assignments[2 * i + i % 2] = policy
        return assignments


def _learning_mask(policy_assignments: List[Optional[Policy]]) -> List[bool]:
    return [p is None for p in policy_assignments]


def _assert_filled_mapped_mask(
    filled_mapped_mask: np.ndarray, mapped_mask: np.ndarray, policy_assignments
) -> None:
    for idx in range(0, len(mapped_mask), 2):
        reset_1, reset_2 = mapped_mask[idx : idx + 2]
        policy_1, policy_2 = policy_assignments[idx : idx + 2]
        if reset_1 != reset_2:
            if policy_1:
                mapped_mask[idx] = True
            elif policy_2:
                mapped_mask[idx + 1] = True
            else:
                raise ValueError(
                    "Expect mapped_mask to be the same for player 1 and 2: "
                    f"{idx}: {reset_1}, {policy_1}; {idx+1}: {reset_2}, {policy_2}"
                )
    assert np.all(mapped_mask == filled_mapped_mask)
