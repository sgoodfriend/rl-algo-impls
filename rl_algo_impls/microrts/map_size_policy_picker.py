import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

import gym.spaces
import numpy as np
import torch
import torch.nn as nn

from rl_algo_impls.runner.config import Config, RunArgs
from rl_algo_impls.runner.running_utils import load_hyperparams, make_policy
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.tensor_utils import NumpyOrDict
from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnv,
    VecEnvObs,
    single_observation_space,
)

MODEL_ROOT_PATH = "rai_microrts_saved_models"

file_path = os.path.abspath(Path(__file__))
root_dir = str(Path(file_path).parent.parent.parent.absolute())


@dataclass
class PickerArgs(RunArgs):
    best: bool = True
    use_paper_obs: bool = False

    def __post_init__(self):
        self._config = None

    @property
    def model_path(self) -> str:
        return os.path.join(
            root_dir, MODEL_ROOT_PATH, self.config.model_dir_name(best=self.best)
        )

    @property
    def config(self) -> Config:
        if self._config is None:
            hyperparams = load_hyperparams(self.algo, self.env)
            self._config = Config(self, hyperparams, root_dir)

        return self._config


errored_on_sizes: Set[int] = set()


class MapSizePolicyPicker(Policy):
    def __init__(
        self,
        picker_args_by_size: Dict[int, PickerArgs],
        env: VecEnv,
        device: torch.device,
    ) -> None:
        super().__init__(env)
        self.to(device)
        self.picker_args_by_size = picker_args_by_size
        self.policies_by_size = {}
        self.policies_by_size_name = nn.ModuleDict(self.policies_by_size)

    def act(
        self,
        obs: VecEnvObs,
        deterministic: bool = True,
        action_masks: Optional[NumpyOrDict] = None,
    ) -> np.ndarray:
        global errored_on_sizes
        obs_space = single_observation_space(self.env)
        assert isinstance(obs_space, gym.spaces.Box)
        obs_size = obs_space.shape[-1]  # type: ignore

        for sz in self.picker_args_by_size:
            if sz >= obs_size:
                if sz > obs_size and sz not in errored_on_sizes:
                    logging.warn(
                        f"Observation size {obs_size} has no matches. Using next largest: {sz}"
                    )
                    errored_on_sizes.add(obs_size)
                return self.use_policy(sz, obs, deterministic, action_masks)
        else:
            sz = max(self.picker_args_by_size)
            if obs_size not in errored_on_sizes:
                logging.error(
                    f"Obseration size {obs_size} exceeded {sz}, using biggest model"
                )
                errored_on_sizes.add(obs_size)
            return self.use_policy(sz, obs, deterministic, action_masks)

    def use_policy(
        self,
        sz: int,
        obs: VecEnvObs,
        deterministic: bool,
        action_masks: Optional[NumpyOrDict],
    ) -> np.ndarray:
        if sz not in self.policies_by_size:
            args = self.picker_args_by_size[sz]
            config = args.config
            assert self.device
            self.policies_by_size[sz] = make_policy(
                config,
                self.env,
                self.device,
                args.model_path,
                **config.policy_hyperparams,
            )
            self.policies_by_size_name = nn.ModuleDict(
                {str(sz): policy for sz, policy in self.policies_by_size.items()}
            )

        policy = self.policies_by_size[sz]
        return policy.act(obs, deterministic=deterministic, action_masks=action_masks)
