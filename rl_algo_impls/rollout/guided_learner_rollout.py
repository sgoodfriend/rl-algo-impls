import logging
from typing import Dict, List, Optional, Sequence, TypeVar

import numpy as np

from rl_algo_impls.rollout.in_process_rollout import InProcessRolloutGenerator
from rl_algo_impls.rollout.trajectory import TrajectoryBuilder
from rl_algo_impls.rollout.trajectory_rollout import TrajectoryRollout
from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.callbacks.summary_wrapper import SummaryWrapper
from rl_algo_impls.shared.data_store.data_store_accessor import (
    AbstractDataStoreAccessor,
)
from rl_algo_impls.shared.data_store.synchronous_data_store_accessor import (
    SynchronousDataStoreAccessor,
)
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.stats import log_scalars
from rl_algo_impls.shared.tensor_utils import (
    NumOrArray,
    NumOrList,
    batch_dict_keys,
    num_or_array,
)
from rl_algo_impls.wrappers.episode_stats_writer import EpisodeStatsWriter
from rl_algo_impls.wrappers.vector_wrapper import find_wrapper


class GuidedLearnerRolloutGenerator(InProcessRolloutGenerator):
    def __init__(
        self,
        config: Config,
        data_store_accessor: AbstractDataStoreAccessor,
        tb_writer: SummaryWrapper,
        guide_policy: ActorCritic,
        switch_range: int,
        gamma: NumOrList = 0.99,
        gae_lambda: NumOrList = 0.95,
        n_steps: int = 2048,
        sde_sample_freq: int = -1,
        scale_advantage_by_values_accuracy: bool = False,
        full_batch_off_accelerator: bool = True,  # Unused, assumed True
        include_logp: bool = True,
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
    ) -> None:
        assert isinstance(data_store_accessor, SynchronousDataStoreAccessor)
        super().__init__(config, data_store_accessor, tb_writer)
        self.gamma = num_or_array(gamma)
        self.gae_lambda = num_or_array(gae_lambda)
        self.guide_policy = guide_policy
        self.guide_policy.eval()

        self.switch_range = switch_range
        self.n_steps = n_steps
        self.sde_sample_freq = sde_sample_freq
        self.scale_advantage_by_values_accuracy = scale_advantage_by_values_accuracy
        if not full_batch_off_accelerator:
            logging.warn(
                f"{self.__class__.__name__} doesn't support full_batch_off_accelerator=False"
            )
        if not include_logp:
            logging.warn(
                f"{self.__class__.__name__} doesn't implement include_logp=False"
            )
        self.subaction_mask = subaction_mask

        self.get_action_mask = getattr(self.vec_env, "get_action_mask", None)

        self.traj_step_by_index = np.zeros(self.vec_env.num_envs, dtype=np.int32)
        self.switch_step_by_index = np.random.randint(
            0, self.switch_range, self.vec_env.num_envs, dtype=np.int32
        )
        self.policies_by_index = [
            self.guide_policy if switch_step > 0 else self.learning_policy
            for switch_step in self.switch_step_by_index
        ]

        self.next_obs = np.zeros(
            (self.env_spaces.num_envs,)
            + self.env_spaces.single_observation_space.shape,
            dtype=self.env_spaces.single_observation_space.dtype,
        )

        self.episode_stats_writer = find_wrapper(self.vec_env, EpisodeStatsWriter)

    def prepare(self) -> None:
        self.next_obs, _ = self.vec_env.reset()
        self.next_action_masks = (
            self.get_action_mask() if self.get_action_mask else None
        )

    def rollout(self) -> TrajectoryRollout:
        (
            learning_policy,
            rollout_params,
            self.tb_writer.timesteps_elapsed,
        ) = self._data_store_view.update_for_rollout_start()
        self.update_rollout_params(rollout_params)
        log_scalars(self.tb_writer, "charts", rollout_params)

        learning_policy.eval()
        learning_policy.reset_noise()
        self.guide_policy.reset_noise()

        traj_builders = [TrajectoryBuilder() for _ in range(self.vec_env.num_envs)]
        completed_trajectories = []

        goal_steps = self.n_steps * self.vec_env.num_envs
        steps = 0
        s = 0
        while steps < goal_steps:
            # FIXME: sde_sample_freq implementation isn't correct because it assumes the
            # transition from guide to learning policy uses the same noise.
            if self.sde_sample_freq > 0 and s > 0 and s % self.sde_sample_freq == 0:
                learning_policy.reset_noise()
                self.guide_policy.reset_noise()
            s += 1

            obs = self.next_obs
            assert isinstance(
                obs, np.ndarray
            ), f"Expected obs to be np.ndarray, got {type(obs)}"
            action_masks = self.next_action_masks

            policy_indexes = []
            actions = []
            values = []
            logprobs = []
            clamped_actions = []

            for policy in set(p for p in self.policies_by_index):
                policy_matches = [policy == p for p in self.policies_by_index]
                (p_actions, p_values, p_logprobs, p_clamped_actions) = policy.step(
                    obs[policy_matches],
                    action_masks=batch_dict_keys(action_masks[policy_matches])
                    if action_masks is not None
                    else None,
                )
                policy_indexes.extend(
                    [idx for idx, m in enumerate(policy_matches) if m]
                )
                actions.extend(split_actions_by_env(p_actions))
                values.extend(p_values)
                logprobs.extend(p_logprobs)
                clamped_actions.extend(p_clamped_actions)

            actions = rearrange(actions, policy_indexes)
            values = rearrange(values, policy_indexes)
            logprobs = rearrange(logprobs, policy_indexes)
            clamped_actions = rearrange(clamped_actions, policy_indexes)

            if self.episode_stats_writer:
                self.episode_stats_writer.steps_per_step = self.policies_by_index.count(
                    learning_policy
                )
            self.next_obs, rewards, terminations, truncations, _ = self.vec_env.step(
                np.array(clamped_actions)
            )
            dones = terminations | truncations
            self.next_action_masks = (
                self.get_action_mask() if self.get_action_mask else None
            )

            self.traj_step_by_index += 1
            for idx, (traj_builder, traj_step, switch_step, done) in enumerate(
                zip(
                    traj_builders,
                    self.traj_step_by_index,
                    self.switch_step_by_index,
                    dones,
                )
            ):
                if traj_step <= switch_step:
                    if done:
                        self.traj_step_by_index[idx] = 0
                        self.switch_step_by_index[idx] = np.random.randint(
                            self.switch_range
                        )
                    elif traj_step == switch_step:
                        self.policies_by_index[idx] = learning_policy
                    continue
                traj_builder.add(
                    obs=obs[idx],
                    reward=rewards[idx],
                    done=done,
                    value=values[idx],
                    logprob=logprobs[idx],
                    action=actions[idx],
                    action_mask=action_masks[idx] if action_masks is not None else None,
                )
                steps += 1
                if done:
                    self.traj_step_by_index[idx] = 0
                    switch_step = np.random.randint(self.switch_range)
                    self.switch_step_by_index[idx] = switch_step
                    self.policies_by_index[idx] = (
                        self.guide_policy if switch_step > 0 else learning_policy
                    )
                    completed_trajectories.append(
                        traj_builder.trajectory(self.gamma, self.gae_lambda)
                    )
                    traj_builder.reset()

        next_values = learning_policy.value(self.next_obs)
        learning_policy.train()

        trajectories = completed_trajectories + [
            traj_builder.trajectory(self.gamma, self.gae_lambda, next_values=next_value)
            for traj_builder, next_value in zip(traj_builders, next_values)
            if len(traj_builder) > 0
        ]
        # Release TrajectoryBuilders because they can be holding a lot of memory.
        traj_builders = None
        return TrajectoryRollout(
            learning_policy.device,
            trajectories,
            scale_advantage_by_values_accuracy=self.scale_advantage_by_values_accuracy,
            subaction_mask=self.subaction_mask,
            action_plane_space=getattr(self.vec_env, "action_plane_space", None),
        )


T = TypeVar("T")


def rearrange(lst: List[T], indices: List[int]) -> List[T]:
    return [itm for _, itm in sorted((idx, item) for idx, item in zip(indices, lst))]


ND = TypeVar("ND", np.ndarray, Dict[str, np.ndarray])


def split_actions_by_env(actions: ND) -> Sequence[ND]:
    if isinstance(actions, dict):
        return [
            {k: v[idx] for k, v in actions.items()}
            for idx in range(len(next(iter(actions.values()))))
        ]
    else:
        return actions  # type: ignore
