from typing import Dict, Optional

import numpy as np
import torch

from rl_algo_impls.rollout.rollout import Rollout
from rl_algo_impls.rollout.sync_step_rollout import SyncStepRolloutGenerator, fold_in
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.tensor_utils import (
    NumOrArray,
    NumpyOrDict,
    TensorOrDict,
    batch_dict_keys,
    tensor_to_numpy,
)
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv


class ReferenceAIRolloutGenerator(SyncStepRolloutGenerator):
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

            (_, self.values[s], logprobs, clamped_policy_actions) = self.policy.step(
                self.next_obs, action_masks=self.next_action_masks
            )
            if self.include_logp:
                self.logprobs = logprobs

            (
                self.next_obs,
                self.rewards[s],
                self.next_episode_starts,
                _,
            ) = self.vec_env.step(clamped_policy_actions)
            actions = batch_dict_keys(getattr(self.vec_env, "last_action"))
            fold_in(self.actions, actions, s)

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

    def actions_to_tensor(self, a: NumpyOrDict) -> TensorOrDict:
        if isinstance(a, dict):
            return {k: torch.as_tensor(v).to(self.policy.device) for k, v in a.items()}
        return torch.as_tensor(a).to(self.policy.device)
