from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from typing import Optional, Sequence

from gym.spaces import Box, Discrete
from shared.policy.on_policy import ActorCritic, default_hidden_sizes


class PPOActorCritic(ActorCritic):
    def __init__(
        self,
        env: VecEnv,
        pi_hidden_sizes: Optional[Sequence[int]] = None,
        v_hidden_sizes: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> None:
        pi_hidden_sizes = (
            pi_hidden_sizes
            if pi_hidden_sizes is not None
            else default_hidden_sizes(env.observation_space)
        )
        v_hidden_sizes = (
            v_hidden_sizes
            if v_hidden_sizes is not None
            else default_hidden_sizes(env.observation_space)
        )
        super().__init__(
            env,
            pi_hidden_sizes,
            v_hidden_sizes,
            **kwargs,
        )
