import numpy as np

from rl_algo_impls.ppo.rollout import Rollout, RolloutGenerator
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnv,
    single_action_space,
    single_observation_space,
)


class SyncStepRolloutGenerator(RolloutGenerator):
    def __init__(
        self, n_steps: int, sde_sample_freq: int, policy: ActorCritic, vec_env: VecEnv
    ) -> None:
        super().__init__(n_steps, sde_sample_freq)
        self.policy = policy
        self.vec_env = vec_env
        self.get_action_mask = getattr(vec_env, "get_action_mask", None)

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
        self.actions = np.zeros(epoch_dim + act_shape, dtype=act_space.dtype)  # type: ignore
        self.rewards = np.zeros(epoch_dim + value_shape, dtype=np.float32)
        self.episode_starts = np.zeros(epoch_dim + value_shape, dtype=np.bool_)
        self.values = np.zeros(epoch_dim, dtype=np.float32)
        self.logprobs = np.zeros(epoch_dim, dtype=np.float32)
        self.action_masks = (
            np.zeros(
                (self.n_steps,) + self.next_action_masks.shape,
                dtype=self.next_action_masks.dtype,
            )
            if self.next_action_masks is not None
            else None
        )

    def rollout(self) -> Rollout:
        self.policy.eval()
        self.policy.reset_noise()
        for s in range(self.n_steps):
            if self.sde_sample_freq > 0 and s > 0 and s % self.sde_sample_freq == 0:
                self.policy.reset_noise()

            self.obs[s] = self.next_obs
            self.episode_starts[s] = self.next_episode_starts
            if self.action_masks is not None:
                self.action_masks[s] = self.next_action_masks

            (
                self.actions[s],
                self.values[s],
                self.logprobs[s],
                clamped_action,
            ) = self.policy.step(self.next_obs, action_masks=self.next_action_masks)
            (
                self.next_obs,
                self.rewards[s],
                self.next_episode_starts,
                _,
            ) = self.vec_env.step(clamped_action)
            self.next_action_masks = (
                self.get_action_mask() if self.get_action_mask else None
            )

        self.policy.train()
        assert isinstance(self.next_obs, np.ndarray)
        return Rollout(
            next_obs=self.next_obs,
            next_action_masks=self.next_action_masks,
            next_episode_starts=self.next_episode_starts,
            obs=self.obs,
            actions=self.actions,
            rewards=self.rewards,
            episode_starts=self.episode_starts,
            values=self.values,
            logprobs=self.logprobs,
            action_masks=self.action_masks,
            total_steps=self.n_steps * self.vec_env.num_envs,
        )
