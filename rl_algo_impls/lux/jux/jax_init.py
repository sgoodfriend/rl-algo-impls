import logging
import os
import platform
import sys

if "jax" in sys.modules:
    logging.warn("jax must be imported after jax_init")


def jax_init():
    if platform.system() == "Darwin":
        if os.environ.get("JAX_PLATFORM_NAME") != "cpu":
            logging.info(
                "JUX doesn't support acceleration on Mac. Setting JAX_PLATFORM_NAME to cpu"
            )
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
