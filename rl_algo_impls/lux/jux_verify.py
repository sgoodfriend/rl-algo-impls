import os

from rl_algo_impls.wrappers.vector_wrapper import VectorEnv


def jux_verify_enabled(env: VectorEnv) -> bool:
    if not jux_verify_flag():
        return False

    from rl_algo_impls.lux.vec_env.vec_lux_env import VecLuxEnv

    return isinstance(env.unwrapped, VecLuxEnv)


def jux_verify_flag() -> bool:
    return os.getenv("JUX_VERIFY", "0") == "1"
