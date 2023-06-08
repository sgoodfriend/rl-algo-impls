import logging
import time
from contextlib import contextmanager


@contextmanager
def measure_time(name: str, threshold_ms: int = 100):
    perf_start = time.perf_counter()
    process_start = time.process_time()
    try:
        yield
    finally:
        perf_ms = (time.perf_counter() - perf_start) * 1000
        process_ms = (time.process_time() - process_start) * 1000
        if perf_ms >= threshold_ms:
            logging.warn(f"{name}: Perf time over {threshold_ms}ms: {int(perf_ms)}ms")
            logging.warn(f"  - {name}: Process time: {int(process_ms)}ms")
