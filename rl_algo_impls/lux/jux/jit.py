import functools
import logging
import platform

from rl_algo_impls.lux.jux.jax_init import jax_init

jax_init()

import jax

JIT_ENABLED = True
if not JIT_ENABLED:
    logging.warn("JIT is disabled. This will be slow.")
    assert platform.system() == "Darwin", f"Only allowed to disable JIT on Mac"


def toggleable_jit(func=None, **jit_kwargs):
    if not JIT_ENABLED:
        return func
    if func is None:
        return functools.partial(jax.jit, **jit_kwargs)
    return jax.jit(func, **jit_kwargs)
