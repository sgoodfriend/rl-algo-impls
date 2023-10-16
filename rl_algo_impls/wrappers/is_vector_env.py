from typing import Any

from rl_algo_impls.wrappers.vector_wrapper import VectorWrapper


class IsVectorEnv(VectorWrapper):
    """
    Override to set properties to match gymnasium.vector.VectorEnv
    """

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.is_vector_env = True
