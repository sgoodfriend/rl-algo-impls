from typing import List, Optional

from rl_algo_impls.wrappers.vectorable_wrapper import VectorableWrapper


class NoopEnvSeed(VectorableWrapper):
    """
    Wrapper to stop a seed call going to the underlying environment.
    """

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        return None
