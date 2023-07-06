import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from rl_algo_impls.runner.config import Config, RunArgs
from rl_algo_impls.runner.running_utils import load_hyperparams, make_policy
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.tensor_utils import NumpyOrDict
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv, VecEnvObs

MODEL_ROOT_PATH = "rai_microrts_saved_models"
PRE_GAME_ANALYSIS_BUFFER_MILLISECONDS = 250
MAX_MILLISECONDS = 75

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
PolicyName = Union[str, int]


class MapSizePolicyPicker(Policy):
    logger = logging.getLogger("MapSizePolicyPicker")

    def __init__(
        self,
        picker_args_by_size: Dict[int, PickerArgs],
        picker_args_by_terrain_md5: Dict[str, PickerArgs],
        env: VecEnv,
        device: torch.device,
        envs_per_size: Dict[int, VecEnv],
        envs_by_terrain_md5: Dict[str, VecEnv],
    ) -> None:
        super().__init__(env)
        self.to(device)

        def load_policy(args: PickerArgs, vec_env: VecEnv) -> Policy:
            policy_hyperparams = args.config.policy_hyperparams
            if "load_run_path" in policy_hyperparams:
                del policy_hyperparams["load_run_path"]
            if "load_run_path_best" in policy_hyperparams:
                del policy_hyperparams["load_run_path_best"]
            return make_policy(
                args.config,
                vec_env,
                device,
                args.model_path,
                **policy_hyperparams,
            )

        self.policies_by_size = {}
        for sz, args in picker_args_by_size.items():
            self.policies_by_size[sz] = load_policy(args, envs_per_size[sz])
        self.policies_by_size_name = nn.ModuleDict(
            {str(sz): p for sz, p in self.policies_by_size.items()}
        )

        policies_by_terrain_md5 = {}
        for terrain_md5, args in picker_args_by_terrain_md5.items():
            policies_by_terrain_md5[terrain_md5] = load_policy(
                args, envs_by_terrain_md5[terrain_md5]
            )
        self.policies_by_terrain_md5 = nn.ModuleDict(policies_by_terrain_md5)

        self.envs_by_policy_name = {**envs_per_size, **envs_by_terrain_md5}

        self.selected_policy_for_terrain_md5: Dict[str, Policy] = {}

        self.last_used: Optional[PolicyName] = None

    def terrain_md5(self) -> str:
        get_terrain_md5 = getattr(self.env, "terrain_md5")
        terrain_md5s = get_terrain_md5()
        return terrain_md5s[0]

    def _valid_policies(self, obs: VecEnvObs) -> List[Tuple[PolicyName, Policy]]:
        global errored_on_sizes

        valid_policies: List[Tuple[PolicyName, Policy]] = []

        terrain_md5 = self.terrain_md5()
        if terrain_md5 in self.policies_by_terrain_md5:
            p = self.policies_by_terrain_md5[terrain_md5]
            assert isinstance(p, Policy)
            valid_policies.append((terrain_md5, p))

        assert isinstance(obs, np.ndarray)
        obs_size = obs.shape[-1]
        for sz in self.policies_by_size:
            if sz >= obs_size:
                valid_policies.append((sz, self.policies_by_size[sz]))
                break
        else:
            sz = max(self.policies_by_size)
            if obs_size not in errored_on_sizes:
                self.logger.error(
                    f"Obseration size {obs_size} exceeded {sz}, using biggest model"
                )
                errored_on_sizes.add(obs_size)
            valid_policies.append((sz, self.policies_by_size[sz]))

        assert valid_policies, f"Found no valid policies for map size {obs_size}"
        return valid_policies

    def act(
        self,
        obs: VecEnvObs,
        deterministic: bool = True,
        action_masks: Optional[NumpyOrDict] = None,
    ) -> np.ndarray:
        terrain_md5 = self.terrain_md5()
        if terrain_md5 in self.selected_policy_for_terrain_md5:
            return self.selected_policy_for_terrain_md5[terrain_md5].act(
                obs, deterministic=deterministic, action_masks=action_masks
            )
        else:
            valid_policies = self._valid_policies(obs)
            policy_name, policy = valid_policies[0]
            return self.use_policy(
                policy, policy_name, obs, deterministic, action_masks
            )

    def pre_game_analysis(
        self,
        obs: VecEnvObs,
        deterministic: bool = True,
        action_masks: Optional[NumpyOrDict] = None,
    ) -> np.ndarray:
        valid_policies = self._valid_policies(obs)
        pga_expiration = getattr(self.env, "pre_game_analysis_expiration_ms")
        observation_by_policy_name: Dict[
            PolicyName, Tuple[VecEnvObs, Optional[NumpyOrDict]]
        ] = {}
        for p_name, p in valid_policies:
            p_env = self.envs_by_policy_name[p_name]
            observation_by_policy_name[p_name] = (
                p_env.reset(),
                getattr(p_env, "get_action_mask")(),
            )

        idx = 0
        ms_by_policy_name: DefaultDict[PolicyName, List[float]] = defaultdict(list)
        start_ms = time.perf_counter() * 1000
        actions_by_policy_name: Dict[PolicyName, np.ndarray] = {}
        # Run each model at least once
        while len(ms_by_policy_name) != len(valid_policies) or (
            start_ms + PRE_GAME_ANALYSIS_BUFFER_MILLISECONDS < pga_expiration
            and any(len(ms) < 100 for ms in ms_by_policy_name.values())
        ):
            p_name, p = valid_policies[idx % len(valid_policies)]
            p_obs, p_action_mask = observation_by_policy_name[p_name]
            actions_by_policy_name[p_name] = p.act(
                p_obs, deterministic=deterministic, action_masks=p_action_mask
            )
            end_ms = time.perf_counter() * 1000
            ms_by_policy_name[p_name].append(end_ms - start_ms)
            start_ms = end_ms
            idx += 1
        avg_ms_by_policy_name = [
            np.median(ms_by_policy_name[p_name]) for p_name, _ in valid_policies
        ]
        for (p_name, p), avg_ms in zip(valid_policies, avg_ms_by_policy_name):
            if avg_ms < MAX_MILLISECONDS:
                policy_name = p_name
                policy = p
                break
        else:
            policy_name, policy = valid_policies[np.argmin(avg_ms_by_policy_name)]
        self.logger.info(f"Pre-game Analysis: Selected {policy_name}")
        for (p_name, p), avg_ms in zip(valid_policies, avg_ms_by_policy_name):
            self.logger.info(
                f"{p_name}: {avg_ms:.1f} ms ({len(ms_by_policy_name[p_name])})"
            )
        self.selected_policy_for_terrain_md5[self.terrain_md5()] = policy
        set_space_transform = getattr(self.env, "set_space_transform")
        if isinstance(policy_name, int):
            set_space_transform(sz=policy_name)
        elif isinstance(policy_name, str):
            set_space_transform(terrain_md5=policy_name)
        else:
            raise ValueError(
                f"Policy name {policy_name} is neither size or terrain_md5"
            )
        return actions_by_policy_name[policy_name]

    def use_policy(
        self,
        policy: Policy,
        policy_name: PolicyName,
        obs: VecEnvObs,
        deterministic: bool,
        action_masks: Optional[NumpyOrDict],
    ) -> np.ndarray:
        if self.last_used != policy_name:
            self.logger.info(f"Using policy {policy_name}")
            self.last_used = policy_name
        return policy.act(obs, deterministic=deterministic, action_masks=action_masks)
