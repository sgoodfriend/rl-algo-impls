from typing import Optional

import gymnasium


class NoopEnvSeed(gymnasium.Wrapper):
    """
    Wrapper to stop a seed call going to the underlying environment.
    """

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        return super().reset(**kwargs)
