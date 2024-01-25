import logging
import os

import torch
import torch.multiprocessing as mp
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO, handlers=[])


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )

    # Check if the process group is initialized correctly
    if torch.distributed.is_initialized():
        logging.info(
            f"Process Group Initialized: Rank {rank}, World Size {world_size}, GPUs: {torch.cuda.device_count()}"
        )
    else:
        logging.info("Failed to initialize Process Group")

    accelerator = Accelerator()
    logging.info(f"accelerator.state: {accelerator.state}")
    logging.info(f"accelerator.device: {accelerator.device}")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    logging.info(f"Spawning {world_size} processes...")
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
