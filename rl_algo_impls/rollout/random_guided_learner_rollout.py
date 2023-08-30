import gc
import logging
from typing import Dict, Optional

import numpy as np

from rl_algo_impls.rollout.discrete_skips_trajectory_builder import (
    DiscreteSkipsTrajectoryBuilder,
)
from rl_algo_impls.rollout.guided_learner_rollout import split_actions_by_env
from rl_algo_impls.rollout.rollout import Rollout, RolloutGenerator
from rl_algo_impls.rollout.trajectory_rollout import TrajectoryRollout
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.tensor_utils import NumOrArray, batch_dict_keys
from rl_algo_impls.wrappers.episode_stats_writer import EpisodeStatsWriter
from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnv,
    find_wrapper,
    single_action_space,
)


class RandomGuidedLearnerRolloutGenerator(RolloutGenerator):
    def __init__(
        self,
        learning_policy: ActorCritic,
        vec_env: VecEnv,
        guide_policy: ActorCritic,
        guide_probability: float,
        n_steps: int = 2048,
        sde_sample_freq: int = -1,
        scale_advantage_by_values_accuracy: bool = False,
        full_batch_off_accelerator: bool = True,  # Unused, assumed True
        include_logp: bool = True,
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
    ) -> None:
        super().__init__()
        self.learning_policy = learning_policy
        self.guide_policy = guide_policy

        self.vec_env = vec_env
        self.guide_probability = guide_probability
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

        self.get_action_mask = getattr(vec_env, "get_action_mask", None)

        self.next_obs = vec_env.reset()
        self.next_action_masks = (
            self.get_action_mask() if self.get_action_mask else None
        )

        self.episode_stats_writer = find_wrapper(vec_env, EpisodeStatsWriter)

    def rollout(self, gamma: NumOrArray, gae_lambda: NumOrArray) -> Rollout:
        self.learning_policy.eval()
        self.learning_policy.reset_noise()
        self.guide_policy.eval()
        self.guide_policy.reset_noise()

        num_envs = self.vec_env.num_envs
        traj_builders = [DiscreteSkipsTrajectoryBuilder() for _ in range(num_envs)]
        completed_trajectories = []

        step_values = np.zeros(
            (num_envs,) + self.learning_policy.value_shape, dtype=np.float32
        )
        step_logprobs = np.zeros((num_envs,), dtype=np.float32)
        act_shape = self.learning_policy.action_shape
        if isinstance(act_shape, dict):
            step_actions = np.zeros((num_envs,), dtype=np.object_)
            step_clamped_actions = np.zeros((num_envs,), dtype=np.object_)
        else:
            act_space = single_action_space(self.vec_env)
            step_actions = np.zeros((num_envs,) + act_shape, dtype=act_space.dtype)
            step_clamped_actions = np.zeros(
                (num_envs,) + act_shape, dtype=act_space.dtype
            )

        goal_steps = self.n_steps * num_envs
        steps = 0
        s = 0
        while steps < goal_steps:
            # FIXME: sde_sample_freq implementation isn't correct because it assumes the
            # transition from guide to learning policy uses the same noise.
            if self.sde_sample_freq > 0 and s > 0 and s % self.sde_sample_freq == 0:
                self.learning_policy.reset_noise()
                self.guide_policy.reset_noise()
            s += 1

            obs = self.next_obs
            assert isinstance(
                obs, np.ndarray
            ), f"Expected obs to be np.ndarray, got {type(obs)}"
            action_masks = self.next_action_masks

            use_guide_policy = np.random.rand(num_envs) < self.guide_probability
            for indices, policy in (
                (use_guide_policy, self.guide_policy),
                (~use_guide_policy, self.learning_policy),
            ):
                if not np.any(indices):
                    continue
                (
                    actions,
                    step_values[indices],
                    step_logprobs[indices],
                    step_clamped_actions[indices],
                ) = policy.step(
                    obs[indices],
                    action_masks=batch_dict_keys(action_masks[indices])
                    if action_masks is not None
                    else None,
                )
                step_actions[indices] = split_actions_by_env(actions)

            if self.episode_stats_writer:
                self.episode_stats_writer.steps_per_step = np.sum(
                    ~use_guide_policy
                ).item()
            self.next_obs, rewards, dones, _ = self.vec_env.step(step_clamped_actions)
            self.next_action_masks = (
                self.get_action_mask() if self.get_action_mask else None
            )

            for idx, (traj_builder, done, is_guided) in enumerate(
                zip(traj_builders, dones, use_guide_policy)
            ):
                if is_guided:
                    traj_builder.step_no_add(done)
                else:
                    steps += 1
                    traj_builder.step_add(
                        obs[idx],
                        rewards[idx],
                        done,
                        step_values[idx],
                        step_logprobs[idx],
                        step_actions[idx],
                        action_masks[idx] if action_masks is not None else None,
                    )
                if done:
                    if len(traj_builder) > 0:
                        completed_trajectories.append(
                            traj_builder.trajectory(gamma, gae_lambda)
                        )
                    traj_builder.reset()
            gc.collect()

        next_values = self.learning_policy.value(self.next_obs)
        self.learning_policy.train()
        self.guide_policy.train()

        trajectories = completed_trajectories + [
            traj_builder.trajectory(gamma, gae_lambda, next_values=next_value)
            for traj_builder, next_value in zip(traj_builders, next_values)
            if len(traj_builder) > 0
        ]
        traj_builders = None
        return TrajectoryRollout(
            trajectories,
            scale_advantage_by_values_accuracy=self.scale_advantage_by_values_accuracy,
            subaction_mask=self.subaction_mask,
            action_plane_space=getattr(self.vec_env, "action_plane_space", None),
        )
