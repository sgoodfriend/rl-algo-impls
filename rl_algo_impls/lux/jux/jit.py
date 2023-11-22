import functools
import logging
import os

from rl_algo_impls.lux.jux.jax_init import jax_init

jax_init()

import jax

JIT_DISABLED = os.getenv("JUX_JIT_DISABLED", "0") == "1"
if JIT_DISABLED:
    logging.warn("JIT is disabled. This will be slow.")


def toggleable_jit(func=None, **jit_kwargs):
    if JIT_DISABLED:
        return func
    if func is None:
        return functools.partial(jax.jit, **jit_kwargs)
    return jax.jit(func, **jit_kwargs)
