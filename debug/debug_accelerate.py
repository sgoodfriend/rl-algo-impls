import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO, handlers=[])

logging.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

accelerator = Accelerator()

logging.info(f"accelerator.state: {accelerator.state}")
