import logging
from typing import Dict, Optional, Type

import numpy as np
from gymnasium.spaces import Dict as DictSpace

from rl_algo_impls.rollout.rollout import Rollout
from rl_algo_impls.rollout.synchronous_rollout_generator import (
    SynchronousRolloutGenerator,
)
from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.data_store.abstract_data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.policy.abstract_policy import AbstractPolicy
from rl_algo_impls.shared.stats import log_scalars
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.shared.tensor_utils import (
    NumOrList,
    batch_dict_keys,
    num_or_array,
    set_items,
)
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces
from rl_algo_impls.wrappers.episode_stats_writer import EpisodeStatsWriter
from rl_algo_impls.wrappers.vector_wrapper import find_wrapper


class SyncStepRolloutGenerator(SynchronousRolloutGenerator):
    def __init__(
        self,
        config: Config,
        data_store_accessor: AbstractDataStoreAccessor,
        tb_writer: AbstractSummaryWrapper,
        rollout_cls: Type[Rollout],
        gamma: NumOrList = 0.99,
        gae_lambda: NumOrList = 0.95,
        n_steps: int = 2048,
        sde_sample_freq: int = -1,
        full_batch_off_accelerator: bool = False,
        include_logp: bool = True,
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
        num_envs_reset_every_rollout: int = 0,
        rolling_num_envs_reset_every_rollout: int = 0,
        random_num_envs_reset_every_rollout: int = 0,
        prepare_steps: int = 0,
        rolling_num_envs_reset_every_prepare_step: int = 0,
    ) -> None:
        super().__init__(config, data_store_accessor, tb_writer, rollout_cls)
        self.gamma = num_or_array(gamma)
        self.gae_lambda = num_or_array(gae_lambda)
        self.n_steps = n_steps
        self.sde_sample_freq = sde_sample_freq
        self.full_batch_off_accelerator = full_batch_off_accelerator
        self.include_logp = include_logp
        self.subaction_mask = subaction_mask

        assert (
            num_envs_reset_every_rollout % 2 == 0
        ), f"num_envs_reset_every_rollout must be even, got {num_envs_reset_every_rollout}"
        self.num_envs_reset_every_rollout = num_envs_reset_every_rollout

        assert (
            rolling_num_envs_reset_every_rollout % 2 == 0
        ), f"rolling_num_envs_reset_every_rollout must be even, got {rolling_num_envs_reset_every_rollout}"
        self.rolling_num_envs_reset_every_rollout = rolling_num_envs_reset_every_rollout
        self.rolling_mask_idx = 0
        self.rolling_reset_indexes = np.random.permutation(self.vec_env.num_envs // 2)

        assert (
            random_num_envs_reset_every_rollout % 2 == 0
        ), f"random_num_envs_reset_every_rollout must be even, got {random_num_envs_reset_every_rollout}"
        self.random_num_envs_reset_every_rollout = random_num_envs_reset_every_rollout
        assert self.vec_env.num_envs > (
            self.num_envs_reset_every_rollout
            + self.rolling_num_envs_reset_every_rollout
            + self.random_num_envs_reset_every_rollout
        ), f"num_envs_reset_every_rollout + rolling_num_envs_reset_every_rollout + random_num_envs_reset_every_rollout must be less than or equal to num_envs, got {self.num_envs_reset_every_rollout + self.rolling_num_envs_reset_every_rollout + self.random_num_envs_reset_every_rollout} > {self.vec_env.num_envs}"

        if (
            rolling_num_envs_reset_every_rollout > 0
            or random_num_envs_reset_every_rollout > 0
            or rolling_num_envs_reset_every_prepare_step > 0
        ):
            assert (
                self.vec_env.num_envs % 2 == 0
            ), f"num_envs must be even for rolling_num_envs_reset_every_rollout, random_num_envs_reset_every_rollout, or rolling_num_envs_reset_every_prepare_step, got {self.vec_env.num_envs}"

        self.prepare_steps = prepare_steps
        self.rolling_num_envs_reset_every_prepare_step = (
            rolling_num_envs_reset_every_prepare_step
        )
        assert (
            rolling_num_envs_reset_every_prepare_step % 2 == 0
        ), f"rolling_num_envs_reset_every_prepare_step must be even, got {rolling_num_envs_reset_every_prepare_step}"

        self.get_action_mask = getattr(self.vec_env, "get_action_mask", None)
        if self.get_action_mask:
            _get_action_mask = self.get_action_mask
            self.get_action_mask = lambda: batch_dict_keys(_get_action_mask())

        (
            obs_space,
            act_space,
            _,
            act_shape,
            num_envs,
            value_shape,
        ) = EnvSpaces.from_vec_env(self.vec_env)

        epoch_dim = (self.n_steps, num_envs)
        step_dim = (num_envs,)

        self.next_obs = np.zeros(step_dim + obs_space.shape, dtype=obs_space.dtype)
        self.next_episode_starts = np.full(step_dim, True, dtype=np.bool_)

        self.obs = np.zeros(epoch_dim + obs_space.shape, dtype=obs_space.dtype)  # type: ignore
        self.rewards = np.zeros(epoch_dim + value_shape, dtype=np.float32)
        self.episode_starts = np.zeros(epoch_dim, dtype=np.bool_)
        self.values = np.zeros(epoch_dim + value_shape, dtype=np.float32)
        self.logprobs = (
            np.zeros(epoch_dim, dtype=np.float32) if self.include_logp else None
        )

        if isinstance(act_shape, dict):
            assert isinstance(act_space, DictSpace)
            self.actions = {
                k: np.zeros(epoch_dim + a_shape, dtype=act_space[k].dtype)
                for k, a_shape in act_shape.items()
            }
        else:
            self.actions = np.zeros(epoch_dim + act_shape, dtype=act_space.dtype)  # type: ignore

    def prepare(self) -> None:
        rollout_view = self.data_store_view.update_for_rollout_start()
        assert rollout_view is not None
        (policy, rollout_params, timesteps_elapsed, _) = rollout_view
        self.tb_writer.on_timesteps_elapsed(timesteps_elapsed)
        self.update_rollout_params(rollout_params)

        self.next_obs, _ = self.vec_env.reset()
        self.next_action_masks = (
            self.get_action_mask() if self.get_action_mask else None
        )
        if isinstance(self.next_action_masks, dict):
            self.action_masks = {
                k: np.zeros(
                    (self.n_steps,) + v.shape,
                    dtype=v.dtype,
                )
                for k, v in self.next_action_masks.items()
            }
        elif self.next_action_masks is not None:
            self.action_masks = np.zeros(
                (self.n_steps,) + self.next_action_masks.shape,
                dtype=self.next_action_masks.dtype,
            )
        else:
            self.action_masks = None

        if not self.prepare_steps:
            return
        from tqdm import tqdm

        logging.info(f"Preparing rollout generation for {self.prepare_steps} steps")
        episode_stats_writer = find_wrapper(self.vec_env, EpisodeStatsWriter)
        if episode_stats_writer:
            episode_stats_writer.disable_record_stats()

        for _ in tqdm(range(0, self.prepare_steps, self.n_steps)):
            self._rollout(policy, output_next_values=False)
            self._reset_envs(
                0,
                self.rolling_num_envs_reset_every_prepare_step,
                0,
            )
        if episode_stats_writer:
            episode_stats_writer.enable_record_stats()

    def rollout(self) -> Optional[Rollout]:
        rollout_view = self.data_store_view.update_for_rollout_start()
        if rollout_view is None:
            return None
        (policy, rollout_params, timesteps_elapsed, _) = rollout_view
        self.tb_writer.on_timesteps_elapsed(timesteps_elapsed)
        self.update_rollout_params(rollout_params)
        log_scalars(self.tb_writer, "charts", rollout_params)

        next_values = self._rollout(policy, output_next_values=True)
        assert next_values is not None

        self._reset_envs(
            self.num_envs_reset_every_rollout,
            self.rolling_num_envs_reset_every_rollout,
            self.random_num_envs_reset_every_rollout,
        )

        return self.rollout_cls(
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

    def _rollout(
        self, policy: AbstractPolicy, output_next_values: bool
    ) -> Optional[np.ndarray]:
        policy.eval()
        policy.reset_noise()
        for s in range(self.n_steps):
            if self.sde_sample_freq > 0 and s > 0 and s % self.sde_sample_freq == 0:
                policy.reset_noise()

            self.obs[s] = self.next_obs
            self.episode_starts[s] = self.next_episode_starts
            if self.action_masks is not None:
                set_items(self.action_masks, self.next_action_masks, s)

            (
                actions,
                self.values[s],
                logprobs,
                clamped_actions,
            ) = policy.step(self.next_obs, action_masks=self.next_action_masks)
            if self.logprobs is not None:
                self.logprobs[s] = logprobs
            set_items(self.actions, actions, s)
            (
                self.next_obs,
                self.rewards[s],
                terminations,
                truncations,
                _,
            ) = self.vec_env.step(clamped_actions)
            self.next_episode_starts = terminations | truncations
            self.next_action_masks = (
                self.get_action_mask() if self.get_action_mask else None
            )

        next_values = policy.value(self.next_obs) if output_next_values else None
        policy.train()
        return next_values

    def _reset_envs(
        self,
        num_envs_reset: int,
        rolling_num_envs_reset: int,
        random_num_envs_reset: int,
    ) -> None:
        assert (
            bool(num_envs_reset)
            + bool(rolling_num_envs_reset)
            + bool(random_num_envs_reset)
            <= 1
        ), "Only one of num_envs_reset, rolling_num_envs_reset, random_num_envs_reset can be set"
        masked_reset_mask = np.zeros(self.vec_env.num_envs, dtype=np.bool_)
        if num_envs_reset > 0:
            masked_reset_mask[-num_envs_reset:] = True
        if rolling_num_envs_reset > 0:
            end_idx = (self.rolling_mask_idx + rolling_num_envs_reset // 2) % len(
                self.rolling_reset_indexes
            )
            if end_idx < self.rolling_mask_idx:
                indexes = np.concatenate(
                    (
                        self.rolling_reset_indexes[self.rolling_mask_idx :],
                        self.rolling_reset_indexes[:end_idx],
                    )
                )
                self.rolling_reset_indexes = np.random.permutation(
                    self.vec_env.num_envs // 2
                )
            else:
                indexes = self.rolling_reset_indexes[self.rolling_mask_idx : end_idx]
            rolling_reset_mask = np.zeros(self.vec_env.num_envs // 2, dtype=np.bool_)
            rolling_reset_mask[indexes] = True
            masked_reset_mask[rolling_reset_mask.repeat(2)] = True
            self.rolling_mask_idx = end_idx
        if random_num_envs_reset > 0:
            pairs_mask = np.zeros(self.vec_env.num_envs // 2, dtype=np.bool_)
            pairs_mask[
                np.random.choice(
                    pairs_mask.shape[0],
                    random_num_envs_reset // 2,
                    replace=False,
                )
            ] = True
            masked_reset_mask[pairs_mask.repeat(2)] = True

        assert masked_reset_mask.sum() == (
            num_envs_reset + rolling_num_envs_reset + random_num_envs_reset
        ), f"Expected {num_envs_reset + rolling_num_envs_reset + random_num_envs_reset} masked resets, got {masked_reset_mask.sum()}"

        if masked_reset_mask.any():
            next_obs, action_mask, _ = self.vec_env.masked_reset(masked_reset_mask)  # type: ignore

            assert isinstance(self.next_obs, np.ndarray)
            self.next_obs[masked_reset_mask] = next_obs
            if self.next_action_masks is not None:
                set_items(
                    self.next_action_masks,
                    batch_dict_keys(action_mask),
                    masked_reset_mask,
                )
