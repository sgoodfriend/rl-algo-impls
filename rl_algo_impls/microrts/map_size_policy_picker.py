import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import torch
import torch.nn as nn

from rl_algo_impls.runner.config import Config, RunArgs
from rl_algo_impls.runner.running_utils import load_hyperparams, make_policy
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.tensor_utils import NumpyOrDict
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv, VecEnvObs

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
        envs_per_size: Dict[int, VecEnv],
    ) -> None:
        super().__init__(env)
        self.to(device)
        self.policies_by_size = {}
        for sz, args in picker_args_by_size.items():
            policy_hyperparams = args.config.policy_hyperparams
            if "load_run_path" in policy_hyperparams:
                del policy_hyperparams["load_run_path"]
            if "load_run_path_best" in policy_hyperparams:
                del policy_hyperparams["load_run_path_best"]
            self.policies_by_size[sz] = make_policy(
                args.config,
                envs_per_size[sz],
                device,
                args.model_path,
                **policy_hyperparams,
            )

        self.policies_by_size_name = nn.ModuleDict(
            {str(sz): p for sz, p in self.policies_by_size.items()}
        )

    def act(
        self,
        obs: VecEnvObs,
        deterministic: bool = True,
        action_masks: Optional[NumpyOrDict] = None,
    ) -> np.ndarray:
        global errored_on_sizes
        assert isinstance(obs, np.ndarray)
        obs_size = obs.shape[-1]

        for sz in self.policies_by_size:
            if sz >= obs_size:
                if sz > obs_size and sz not in errored_on_sizes:
                    logging.warn(
                        f"Observation size {obs_size} has no matches. Using next largest: {sz}"
                    )
                    errored_on_sizes.add(obs_size)
                return self.use_policy(sz, obs, deterministic, action_masks)
        else:
            sz = max(self.policies_by_size)
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
        return self.policies_by_size[sz].act(
            obs, deterministic=deterministic, action_masks=action_masks
        )
