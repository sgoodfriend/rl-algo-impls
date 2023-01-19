from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from typing import Sequence

from shared.policy.on_policy import ActorCritic


class VPGActorCritic(ActorCritic):
    def __init__(
        self,
        env: VecEnv,
        hidden_sizes: Sequence[int],
        init_layers_orthogonal: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            env,
            hidden_sizes,
            hidden_sizes,
            init_layers_orthogonal=init_layers_orthogonal,
            **kwargs,
        )
