import gym

from typing import Any


class IsVectorEnv(gym.Wrapper):
    """
    Override to set properties to match gym.vector.VectorEnv
    """

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.is_vector_env = True
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
