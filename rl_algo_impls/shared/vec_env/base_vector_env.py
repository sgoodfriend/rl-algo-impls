import gymnasium
from gymnasium.experimental.vector.utils import batch_space
from gymnasium.experimental.vector.vector_env import VectorEnv


class BaseVectorEnv(VectorEnv):
    def __init__(
        self,
        num_envs: int,
        single_observation_space: gymnasium.Space,
        single_action_space: gymnasium.Space,
    ) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.is_vector_env = True

        self.single_observation_space = single_observation_space
        self.single_action_space = single_action_space

        self.observation_space = batch_space(
            self.single_observation_space, n=self.num_envs
        )
        self.action_space = batch_space(self.single_action_space, n=self.num_envs)
