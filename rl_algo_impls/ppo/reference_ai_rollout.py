import logging

import numpy as np
import torch

from rl_algo_impls.ppo.rollout import Rollout
from rl_algo_impls.ppo.sync_step_rollout import SyncStepRolloutGenerator, fold_in
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.tensor_utils import NumpyOrDict, TensorOrDict, tensor_to_numpy
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv


class ReferenceAIRollout(SyncStepRolloutGenerator):
    def __init__(
        self,
        training_policy: ActorCritic,
        reference_vec_env: VecEnv,
        n_steps: int = 2048,
        sde_sample_freq: int = -1,
    ) -> None:
        super().__init__(training_policy, reference_vec_env, n_steps, sde_sample_freq)
        if isinstance(self.actions, dict):
            self.zero_action = {k: np.zeros_like(v[0]) for k, v in self.actions.items()}
        else:
            self.zero_action = np.zeros_like(self.actions[0])

    def rollout(self) -> Rollout:
        self.policy.eval()
        self.policy.reset_noise()
        for s in range(self.n_steps):
            if self.sde_sample_freq > 0 and s > 0 and s % self.sde_sample_freq == 0:
                self.policy.reset_noise()

            self.obs[s] = self.next_obs
            self.episode_starts[s] = self.next_episode_starts
            if self.action_masks is not None:
                fold_in(self.action_masks, self.next_action_masks, s)

            t_obs = torch.as_tensor(self.next_obs).to(self.policy.device)
            t_action_masks = self.actions_to_tensor(self.next_action_masks)
            (
                self.next_obs,
                self.rewards[s],
                self.next_episode_starts,
                _,
            ) = self.vec_env.step(self.zero_action)
            actions = getattr(self.vec_env, "last_action")
            fold_in(self.actions, actions, s)

            t_actions = self.actions_to_tensor(actions)
            with torch.no_grad():
                (logprobs, _, values) = self.policy(
                    t_obs, t_actions, action_masks=t_action_masks
                )
            self.logprobs[s] = tensor_to_numpy(logprobs)
            self.values[s] = tensor_to_numpy(values)

            self.next_action_masks = (
                self.get_action_mask() if self.get_action_mask else None
            )

        self.policy.train()
        assert isinstance(self.next_obs, np.ndarray)
        return Rollout(
            next_obs=self.next_obs,
            next_episode_starts=self.next_episode_starts,
            obs=self.obs,
            actions=self.actions,
            rewards=self.rewards,
            episode_starts=self.episode_starts,
            values=self.values,
            logprobs=self.logprobs,
            action_masks=self.action_masks,
        )

    def actions_to_tensor(self, a: NumpyOrDict) -> TensorOrDict:
        if isinstance(a, dict):
            return {k: torch.as_tensor(v).to(self.policy.device) for k, v in a.items()}
        return torch.as_tensor(a).to(self.policy.device)
