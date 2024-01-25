import logging
import os

import torch
import torch.multiprocessing as mp

logging.basicConfig(level=logging.INFO, handlers=[])


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )
    from accelerate import Accelerator

    accelerator = Accelerator()

    logging.info(f"accelerator.state: {accelerator.state}")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
