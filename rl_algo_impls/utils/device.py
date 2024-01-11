import logging
import os
import warnings
from typing import List

import torch
import torch.backends
from gymnasium.spaces import Box, Discrete

from rl_algo_impls.runner.config import Config, Hyperparams, RunArgs
from rl_algo_impls.runner.worker_hyperparams import WorkerHyperparams
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces
from rl_algo_impls.shared.vec_env.utils import is_microrts
from rl_algo_impls.utils.ray import get_max_num_threads


def get_device(config: Config, env_spaces: EnvSpaces) -> torch.device:
    device = config.device
    # cuda by default
    if device == "auto":
        device = "cuda"
        # Apple MPS is a second choice (sometimes)
        if device == "cuda" and not torch.cuda.is_available():
            device = "mps"
        # If no MPS, fallback to cpu
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        # Simple environments like Discreet and 1-D Boxes might also be better
        # served with the CPU.
        if device == "mps":
            obs_space = env_spaces.single_observation_space
            if isinstance(obs_space, Discrete):
                device = "cpu"
            elif isinstance(obs_space, Box) and len(obs_space.shape) == 1:
                device = "cpu"
            if is_microrts(config):
                device = "cpu"

    logging.info(f"Device: {device}")
    return torch.device(device)


def initialize_cuda_devices(args: RunArgs, hyperparams: Hyperparams) -> List[int]:
    worker_hyperparams = WorkerHyperparams(**hyperparams.worker_hyperparams)
    if args.device_indexes is not None:
        gpu_ids = args.device_indexes
    else:
        import GPUtil

        gpu_ids_by_avail_memory = GPUtil.getAvailable(
            order="memory",
            maxLoad=1.0,
            maxMemory=1.0,
            limit=worker_hyperparams.desired_num_accelerators,
        )
        gpu_ids = gpu_ids_by_avail_memory[: worker_hyperparams.desired_num_accelerators]
    if gpu_ids:
        if worker_hyperparams.desired_num_accelerators > len(gpu_ids):
            warnings.warn(
                f"Desired {worker_hyperparams.desired_num_accelerators} GPUs but only found {len(gpu_ids)} GPUs"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        import torch

        assert torch.cuda.device_count() == len(
            gpu_ids
        ), f"GPU count mismatch: {torch.cuda.device_count()} vs expected {len(gpu_ids)}"
        torch.set_num_threads(get_max_num_threads())
    elif worker_hyperparams.desired_num_accelerators > 1:
        warnings.warn(
            f"No GPUs found despite desiring {worker_hyperparams.desired_num_accelerators} GPUs"
        )
    return gpu_ids
