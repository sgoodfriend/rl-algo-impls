from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from typing import Optional, Sequence

from gym.spaces import Box, Discrete
from shared.policy.on_policy import ActorCritic


class PPOActorCritic(ActorCritic):
    def __init__(
        self,
        env: VecEnv,
        pi_hidden_sizes: Optional[Sequence[int]] = None,
        v_hidden_sizes: Optional[Sequence[int]] = None,
        init_layers_orthogonal: bool = True,
        **kwargs,
    ) -> None:
        obs_space = env.observation_space
        if isinstance(obs_space, Box):
            if len(obs_space.shape) == 3:
                pi_hidden_sizes = pi_hidden_sizes or [512]
                v_hidden_sizes = v_hidden_sizes or [512]
            elif len(obs_space.shape) == 1:
                pi_hidden_sizes = pi_hidden_sizes or [64, 64]
                v_hidden_sizes = v_hidden_sizes or [64, 64]
            else:
                raise ValueError(f"Unsupported observation space: {obs_space}")
        elif isinstance(obs_space, Discrete):
            pi_hidden_sizes = pi_hidden_sizes or [64]
            v_hidden_sizes = v_hidden_sizes or [64]
        else:
            raise ValueError(f"Unsupported observation space: {obs_space}")
        super().__init__(
            env,
            pi_hidden_sizes,
            v_hidden_sizes,
            init_layers_orthogonal=init_layers_orthogonal,
            **kwargs,
        )
