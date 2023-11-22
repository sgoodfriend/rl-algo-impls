import os


def jux_verify_enabled():
    return os.getenv("JUX_VERIFY", "0") == "1"
