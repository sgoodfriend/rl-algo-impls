import multiprocessing
import os
from typing import Optional


def init_ray_actor(num_threads: Optional[int] = None):
    if num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
    else:
        os.environ.pop("OMP_NUM_THREADS", None)
        num_threads = multiprocessing.cpu_count()
    import torch

    if torch.get_num_threads() != num_threads:
        torch.set_num_threads(num_threads)
