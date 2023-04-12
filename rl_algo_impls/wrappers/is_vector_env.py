from typing import Any

from rl_algo_impls.wrappers.vectorable_wrapper import VectorableWrapper


class IsVectorEnv(VectorableWrapper):
    """
    Override to set properties to match gym.vector.VectorEnv
    """

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.is_vector_env = True
