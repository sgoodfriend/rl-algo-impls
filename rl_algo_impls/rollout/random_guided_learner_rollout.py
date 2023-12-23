import gc
import logging
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from rl_algo_impls.rollout.discrete_skips_trajectory_builder import (
    DiscreteSkipsTrajectoryBuilder,
)
from rl_algo_impls.rollout.guided_learner_rollout import split_actions_by_env
from rl_algo_impls.rollout.in_process_rollout import InProcessRolloutGenerator
from rl_algo_impls.rollout.rollout import Rollout
from rl_algo_impls.rollout.trajectory_rollout import TrajectoryRollout
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.tensor_utils import NumOrArray, batch_dict_keys
from rl_algo_impls.wrappers.episode_stats_writer import EpisodeStatsWriter
from rl_algo_impls.wrappers.vector_wrapper import VectorEnv, find_wrapper


class RandomGuidedLearnerRolloutGenerator(InProcessRolloutGenerator):
    def __init__(
        self,
        learning_policy: ActorCritic,
        vec_env: VectorEnv,
        guide_policy: ActorCritic,
        guide_probability: float,
        n_steps: int = 2048,
        sde_sample_freq: int = -1,
        scale_advantage_by_values_accuracy: bool = False,
        full_batch_off_accelerator: bool = True,  # Unused, assumed True
        include_logp: bool = True,
        subaction_mask: Optional[Dict[int, Dict[int, int]]] = None,
        skip_no_action_steps: bool = False,
        num_envs_reset_every_rollout: int = 0,
    ) -> None:
        super().__init__(learning_policy, vec_env)
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
        self.skip_no_action_steps = skip_no_action_steps
        self.num_envs_reset_every_rollout = num_envs_reset_every_rollout

        self.get_action_mask = getattr(vec_env, "get_action_mask", None)

        self.next_obs = np.zeros(
            (self.env_spaces.num_envs,)
            + self.env_spaces.single_observation_space.shape,
            dtype=self.env_spaces.single_observation_space.dtype,
        )

        self.episode_stats_writer = find_wrapper(vec_env, EpisodeStatsWriter)

        if skip_no_action_steps:
            assert (
                self.get_action_mask is not None
            ), f"skip_no_action_steps requires get_action_mask to be implemented on {vec_env}"
            act_space = vec_env.single_action_space
            act_shape = self.learning_policy.action_shape
            if isinstance(act_shape, dict):
                self.zero_action = np.array(
                    [
                        {
                            k: np.zeros(v, dtype=act_space[k].dtype)
                            for k, v in act_shape.items()
                        }
                        for _ in range(self.num_envs)
                    ]
                )
            else:
                self.zero_action = np.zeros(
                    (self.num_envs,) + act_shape, dtype=act_space.dtype
                )

    @property
    def num_envs(self) -> int:
        return self.vec_env.num_envs

    def prepare(self) -> None:
        self.next_obs, _ = self.vec_env.reset()
        self.next_action_masks = (
            self.get_action_mask() if self.get_action_mask else None
        )

    def rollout(self, gamma: NumOrArray, gae_lambda: NumOrArray) -> Rollout:
        self.learning_policy.eval()
        self.learning_policy.reset_noise()
        self.guide_policy.eval()
        self.guide_policy.reset_noise()

        num_envs = self.num_envs
        traj_builders = [DiscreteSkipsTrajectoryBuilder() for _ in range(num_envs)]
        completed_trajectories = []

        act_shape = self.learning_policy.action_shape
        act_space = self.vec_env.single_action_space
        if isinstance(act_shape, dict):
            step_clamped_actions = np.zeros((num_envs,), dtype=np.object_)
        else:
            step_clamped_actions = np.zeros(
                (num_envs,) + act_shape, dtype=act_space.dtype
            )
        assert isinstance(
            self.next_obs, np.ndarray
        ), f"Expected obs to be np.ndarray, got {type(self.next_obs)}"

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
            action_masks = self.next_action_masks

            use_zero_policy = (
                ~has_actions(action_masks)
                if self.skip_no_action_steps and action_masks is not None
                else np.full(num_envs, False)
            )
            use_learning_policy = ~use_zero_policy & (
                np.random.rand(num_envs) >= self.guide_probability
            )
            use_guide_policy = ~use_zero_policy & ~use_learning_policy
            assert np.all(
                use_zero_policy + use_learning_policy + use_guide_policy == 1
            ), f"Expected exactly one policy to be used, got {use_zero_policy}, {use_learning_policy}, {use_guide_policy}"
            if np.any(use_zero_policy):
                step_clamped_actions[use_zero_policy] = self.zero_action[
                    use_zero_policy
                ]
            if np.any(use_guide_policy):
                (
                    _,
                    _,
                    _,
                    step_clamped_actions[use_guide_policy],
                ) = self.guide_policy.step(
                    obs[use_guide_policy],
                    action_masks=batch_dict_keys(action_masks[use_guide_policy])
                    if action_masks is not None
                    else None,
                )
            if np.any(use_learning_policy):
                # Copy the obs necessary for learning so that the original obs with
                # guide observations can be discarded.
                step_obs = (
                    obs[use_learning_policy].copy()
                    if not np.all(use_learning_policy)
                    else obs
                )
                step_action_masks = (
                    (
                        action_masks[use_learning_policy].copy()
                        if not np.all(use_learning_policy)
                        else action_masks
                    )
                    if action_masks is not None
                    else None
                )
                (
                    actions,
                    step_values,
                    step_logprobs,
                    step_clamped_actions[use_learning_policy],
                ) = self.learning_policy.step(
                    step_obs,
                    action_masks=batch_dict_keys(step_action_masks)
                    if step_action_masks is not None
                    else None,
                )
                step_actions = split_actions_by_env(actions)

            learning_steps = np.sum(use_learning_policy).item()
            if self.episode_stats_writer:
                self.episode_stats_writer.steps_per_step = learning_steps

            self.next_obs, rewards, terminations, truncations, _ = self.vec_env.step(
                step_clamped_actions
            )
            dones = terminations | truncations
            self.next_action_masks = (
                self.get_action_mask() if self.get_action_mask else None
            )

            for idx in np.where(use_guide_policy | use_zero_policy)[0]:
                traj_builders[idx].step_no_add(rewards[idx], dones[idx], gamma)
            steps += learning_steps
            for step_idx, idx in enumerate(np.where(use_learning_policy)[0]):
                traj_builders[idx].step_add(
                    step_obs[step_idx],
                    rewards[idx],
                    dones[idx],
                    step_values[step_idx],
                    step_logprobs[step_idx],
                    step_actions[step_idx],
                    step_action_masks[step_idx]
                    if step_action_masks is not None
                    else None,
                    gamma,
                )
            for traj_builder in traj_builders:
                if traj_builder.done:
                    if len(traj_builder) > 0:
                        completed_trajectories.append(
                            traj_builder.trajectory(gamma, gae_lambda)
                        )
                    traj_builder.reset()

        next_values = self.learning_policy.value(self.next_obs)
        self.learning_policy.train()
        self.guide_policy.train()

        trajectories = completed_trajectories + [
            traj_builder.trajectory(gamma, gae_lambda, next_values=next_value)
            for traj_builder, next_value in zip(traj_builders, next_values)
            if len(traj_builder) > 0
        ]
        traj_builders = None

        if self.num_envs_reset_every_rollout > 0:
            masked_reset_mask = np.zeros(self.num_envs, dtype=np.bool_)
            masked_reset_mask[-self.num_envs_reset_every_rollout :] = True
            next_obs, action_mask, _ = self.vec_env.masked_reset(masked_reset_mask)
            self.next_obs[-self.num_envs_reset_every_rollout :] = next_obs
            if self.next_action_masks is not None:
                self.next_action_masks[
                    -self.num_envs_reset_every_rollout :
                ] = action_mask

        gc.collect()
        return TrajectoryRollout(
            self.policy.device,
            trajectories,
            scale_advantage_by_values_accuracy=self.scale_advantage_by_values_accuracy,
            subaction_mask=self.subaction_mask,
            action_plane_space=getattr(self.vec_env, "action_plane_space", None),
        )


def has_actions(action_mask: np.ndarray) -> NDArray[np.bool_]:
    if isinstance(action_mask[0], dict):
        return np.array([any(m.any() for m in mask.values()) for mask in action_mask])
    return action_mask.any(axis=tuple(range(1, len(action_mask.shape))))
