import gym
import gym.spaces.dict
import numpy as np

from gym import ObservationWrapper


class GetRgbObservation(ObservationWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.dict.Dict)

        self.observation_space = env.observation_space["rgb"]  # type: ignore
        if getattr(env, "is_vector_env"):
            self.single_observation_space = env.single_observation_space["rgb"]  # type: ignore

    def observation(self, observation: gym.spaces.dict.Dict) -> np.ndarray:
        return observation["rgb"]  # type: ignore
