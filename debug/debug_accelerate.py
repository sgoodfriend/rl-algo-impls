import logging

import torch
from accelerate import Accelerator

logging.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

accelerator = Accelerator()

logging.info(f"accelerator.state: {accelerator.state}")
