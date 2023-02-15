import gym

from typing import List, Optional


class NoopEnvSeed(gym.Wrapper):
    """
    Wrapper to stop a seed call going to the underlying environment.
    """

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        return None
