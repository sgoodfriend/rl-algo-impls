from typing import Dict, Optional, TypeVar

import numpy as np

from rl_algo_impls.ppo.rollout import Rollout, RolloutGenerator
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.tensor_utils import NumOrArray, batch_dict_keys
from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnv,
    single_action_space,
    single_observation_space,
)


class SyncStepRolloutGenerator(RolloutGenerator):
    def __init__(
        self,
        policy: ActorCritic,
        vec_env: VecEnv,
        n_steps: int = 2048,
        sde_sample_freq: int = -1,
        scale_advantage_by_values_accuracy: bool = False,
        full_batch_off_accelerator: bool = False,
        include_logp: bool = True,
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.vec_env = vec_env
        self.n_steps = n_steps
        self.sde_sample_freq = sde_sample_freq
        self.scale_advantage_by_values_accuracy = scale_advantage_by_values_accuracy
        self.full_batch_off_accelerator = full_batch_off_accelerator
        self.include_logp = include_logp

        self.get_action_mask = getattr(vec_env, "get_action_mask", None)
        if self.get_action_mask:
            _get_action_mask = self.get_action_mask
            self.get_action_mask = lambda: batch_dict_keys(_get_action_mask())

        epoch_dim = (self.n_steps, vec_env.num_envs)
        step_dim = (vec_env.num_envs,)
        obs_space = single_observation_space(vec_env)
        act_space = single_action_space(vec_env)
        act_shape = self.policy.action_shape
        value_shape = self.policy.value_shape

        self.next_obs = vec_env.reset()
        self.next_action_masks = (
            self.get_action_mask() if self.get_action_mask else None
        )
        self.next_episode_starts = np.full(step_dim, True, dtype=np.bool_)

        self.obs = np.zeros(epoch_dim + obs_space.shape, dtype=obs_space.dtype)  # type: ignore
        self.rewards = np.zeros(epoch_dim + value_shape, dtype=np.float32)
        self.episode_starts = np.zeros(epoch_dim, dtype=np.bool_)
        self.values = np.zeros(epoch_dim + value_shape, dtype=np.float32)
        self.logprobs = (
            np.zeros(epoch_dim, dtype=np.float32) if self.include_logp else None
        )
        self.subaction_mask = subaction_mask

        if isinstance(act_shape, dict):
            self.actions = {
                k: np.zeros(epoch_dim + a_shape, dtype=act_space[k].dtype)
                for k, a_shape in act_shape.items()
            }
            self.action_masks = (
                {
                    k: np.zeros(
                        (self.n_steps,) + v.shape,
                        dtype=v.dtype,
                    )
                    for k, v in self.next_action_masks.items()
                }
                if self.next_action_masks is not None
                else None
            )
        else:
            self.actions = np.zeros(epoch_dim + act_shape, dtype=act_space.dtype)  # type: ignore
            self.action_masks = (
                np.zeros(
                    (self.n_steps,) + self.next_action_masks.shape,
                    dtype=self.next_action_masks.dtype,
                )
                if self.next_action_masks is not None
                else None
            )

    def rollout(self, gamma: NumOrArray, gae_lambda: NumOrArray) -> Rollout:
        self.policy.eval()
        self.policy.reset_noise()
        for s in range(self.n_steps):
            if self.sde_sample_freq > 0 and s > 0 and s % self.sde_sample_freq == 0:
                self.policy.reset_noise()

            self.obs[s] = self.next_obs
            self.episode_starts[s] = self.next_episode_starts
            if self.action_masks is not None:
                fold_in(self.action_masks, self.next_action_masks, s)

            (
                actions,
                self.values[s],
                logprobs,
                clamped_actions,
            ) = self.policy.step(self.next_obs, action_masks=self.next_action_masks)
            if self.logprobs is not None:
                self.logprobs[s] = logprobs
            fold_in(self.actions, actions, s)
            (
                self.next_obs,
                self.rewards[s],
                self.next_episode_starts,
                _,
            ) = self.vec_env.step(clamped_actions)
            self.next_action_masks = (
                self.get_action_mask() if self.get_action_mask else None
            )

        next_values = self.policy.value(self.next_obs)

        self.policy.train()
        assert isinstance(self.next_obs, np.ndarray)
        return Rollout(
            next_episode_starts=self.next_episode_starts,
            next_values=next_values,
            obs=self.obs,
            actions=self.actions,
            rewards=self.rewards,
            episode_starts=self.episode_starts,
            values=self.values,
            logprobs=self.logprobs,
            action_masks=self.action_masks,
            gamma=gamma,
            gae_lambda=gae_lambda,
            scale_advantage_by_values_accuracy=self.scale_advantage_by_values_accuracy,
            full_batch_off_accelerator=self.full_batch_off_accelerator,
            subaction_mask=self.subaction_mask,
            action_plane_space=getattr(self.vec_env, "action_plane_space", None),
        )


ND = TypeVar("ND", np.ndarray, Dict[str, np.ndarray])


def fold_in(destination: ND, subset: ND, idx: int):
    def fn(_d: np.ndarray, _s: np.ndarray):
        _d[idx] = _s

    if isinstance(destination, dict):
        assert isinstance(subset, dict)
        for k, d in destination.items():
            fn(d, subset[k])
    else:
        assert isinstance(subset, np.ndarray)
        fn(destination, subset)
