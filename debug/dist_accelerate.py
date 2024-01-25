import logging

import torch
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)

accelerator = Accelerator()
logging.info(f"accelerator.state: {accelerator.state}")

logging.info(f"torch.distributed.is_available(): {torch.distributed.is_available()}")
logging.info(
    f"torch.distributed.is_initialized(): {torch.distributed.is_initialized()}"
)
if torch.distributed.is_initialized():
    logging.info(f"torch.distributed.get_backend(): {torch.distributed.get_backend()}")
    logging.info(f"torch.distributed.get_rank(): {torch.distributed.get_rank()}")
    logging.info(
        f"torch.distributed.get_world_size(): {torch.distributed.get_world_size()}"
    )

logging.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
logging.info(f"accelerator.device: {accelerator.device}")
