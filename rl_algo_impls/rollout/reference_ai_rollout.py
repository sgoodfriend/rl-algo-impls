from typing import Optional

import numpy as np

from rl_algo_impls.rollout.sync_step_rollout import SyncStepRolloutGenerator, fold_in
from rl_algo_impls.rollout.vec_rollout import VecRollout
from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.stats import log_scalars
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.shared.tensor_utils import batch_dict_keys


class ReferenceAIRolloutGenerator(SyncStepRolloutGenerator):
    def __init__(
        self,
        config: Config,
        data_store_accessor: AbstractDataStoreAccessor,
        tb_writer: AbstractSummaryWrapper,
        **kwargs
    ) -> None:
        super().__init__(config, data_store_accessor, tb_writer, **kwargs)
        if not self.include_logp:
            if isinstance(self.actions, dict):
                self.zero_action = np.array(
                    [
                        {
                            k: np.zeros(v.shape[2:], dtype=v.dtype)
                            for k, v in self.actions.items()
                        }
                        for _ in range(self.vec_env.num_envs)
                    ]
                )
            else:
                self.zero_action = np.zeros(
                    (self.vec_env.num_envs,) + self.actions.shape[2:],
                    dtype=self.actions.dtype,
                )

    def rollout(self) -> Optional[VecRollout]:
        rollout_view = self.data_store_view.update_for_rollout_start()
        if rollout_view is None:
            return None
        (
            policy,
            rollout_params,
            self.tb_writer.timesteps_elapsed,
        ) = rollout_view
        self.update_rollout_params(rollout_params)
        log_scalars(self.tb_writer, rollout_params, self.tb_writer.timesteps_elapsed)

        policy.eval()
        policy.reset_noise()

        for s in range(self.n_steps):
            if self.sde_sample_freq > 0 and s > 0 and s % self.sde_sample_freq == 0:
                policy.reset_noise()

            self.obs[s] = self.next_obs
            self.episode_starts[s] = self.next_episode_starts
            if self.action_masks is not None:
                fold_in(self.action_masks, self.next_action_masks, s)

            if self.include_logp:
                (
                    _,
                    self.values[s],
                    self.logprobs,
                    step_actions,
                ) = policy.step(self.next_obs, action_masks=self.next_action_masks)
            else:
                self.values[s] = policy.value(self.next_obs)
                step_actions = self.zero_action

            (
                self.next_obs,
                self.rewards[s],
                self.next_episode_starts,
                _,
            ) = self.vec_env.step(step_actions)
            actions = batch_dict_keys(getattr(self.vec_env, "last_action"))
            fold_in(self.actions, actions, s)

            self.next_action_masks = (
                self.get_action_mask() if self.get_action_mask else None
            )

        next_values = policy.value(self.next_obs)

        policy.train()
        assert isinstance(self.next_obs, np.ndarray)
        return VecRollout(
            policy.device,
            next_episode_starts=self.next_episode_starts,
            next_values=next_values,
            obs=self.obs,
            actions=self.actions,
            rewards=self.rewards,
            episode_starts=self.episode_starts,
            values=self.values,
            logprobs=self.logprobs,
            action_masks=self.action_masks,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            full_batch_off_accelerator=self.full_batch_off_accelerator,
            subaction_mask=self.subaction_mask,
            action_plane_space=getattr(self.vec_env, "action_plane_space", None),
        )
