import logging

import torch
import torch.backends
from gymnasium.spaces import Box, Discrete

from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces
from rl_algo_impls.shared.vec_env.utils import is_microrts


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
