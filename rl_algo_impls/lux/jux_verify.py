import os


def jux_verify_flag() -> bool:
    return os.getenv("JUX_VERIFY", "0") == "1"
