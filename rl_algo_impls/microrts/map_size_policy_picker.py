import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    DefaultDict,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import torch
import torch.nn as nn

from rl_algo_impls.runner.config import Config, RunArgs
from rl_algo_impls.runner.running_utils import load_hyperparams, make_policy
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.tensor_utils import NumpyOrDict
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv, VecEnvObs

MODEL_ROOT_PATH = "rai_microrts_saved_models"
EXPONENTIAL_MOVING_AVERAGE_SPAN = 100
PRE_GAME_ANALYSIS_BUFFER_MILLISECONDS = 200
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


class ExponentialMovingAverage:
    average: float
    span: int

    def __init__(self, initial_values: Sequence[float], span: int) -> None:
        self.average = np.mean(initial_values[-span:]).item()
        self.span = span

    def add_values(self, values: Sequence[float]) -> float:
        alpha = 2 / (self.span + 1)
        avg = self.average
        for v in values:
            avg = (v * alpha) + (avg * (1 - alpha))
        self.average = avg
        return avg


T = TypeVar("T")


class ExponentialMovingAverages:
    ema_by_key: Dict[PolicyName, ExponentialMovingAverage] = {}
    span: int

    def __init__(self, span: int) -> None:
        self.span = span

    def add(self, key: PolicyName, values: Sequence[float]) -> float:
        if key not in self.ema_by_key:
            self.ema_by_key[key] = ExponentialMovingAverage(values, self.span)
        else:
            self.ema_by_key[key].add_values(values)
        return self.ema_by_key[key].average

    def get(self, key: PolicyName, default_value: T = None) -> Union[float, T]:
        if key in self.ema_by_key:
            return self.ema_by_key[key].average
        else:
            return default_value


class MapSizePolicyPicker(Policy):
    logger = logging.getLogger("MapSizePolicyPicker")
    avg_ms_by_terrain_md5: DefaultDict[str, ExponentialMovingAverages] = defaultdict(
        lambda: ExponentialMovingAverages(EXPONENTIAL_MOVING_AVERAGE_SPAN)
    )
    selected_policy_for_terrain_md5: Dict[str, PolicyName] = {}
    last_used: Optional[PolicyName] = None

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

        self.policies_by_policy_name = {
            **self.policies_by_size,
            **policies_by_terrain_md5,
        }
        self.envs_by_policy_name = {**envs_per_size, **envs_by_terrain_md5}

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
            policy_name = self.selected_policy_for_terrain_md5[terrain_md5]
            policy = self.policies_by_policy_name[policy_name]
        else:
            valid_policies = self._valid_policies(obs)
            policy_name, policy = valid_policies[0]
        return self.use_policy(
            policy, policy_name, terrain_md5, obs, deterministic, action_masks
        )

    def pre_game_analysis(
        self,
        obs: VecEnvObs,
        deterministic: bool = True,
        action_masks: Optional[NumpyOrDict] = None,
    ) -> np.ndarray:
        terrain_md5 = self.terrain_md5()
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
            and any(
                len(ms) < EXPONENTIAL_MOVING_AVERAGE_SPAN + 10
                for ms in ms_by_policy_name.values()
            )
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
        for policy_name, execution_times in ms_by_policy_name.items():
            self.avg_ms_by_terrain_md5[terrain_md5].add(policy_name, execution_times)
        for p_name, _ in valid_policies:
            if (
                self.avg_ms_by_terrain_md5[terrain_md5].get(p_name, 0)
                < MAX_MILLISECONDS
            ):
                selected_policy_name = p_name
                break
        else:
            selected_policy_name, _ = valid_policies[
                np.argmin(
                    [
                        self.avg_ms_by_terrain_md5[terrain_md5].get(p_name, 0)
                        for (p_name, _) in valid_policies
                    ]
                )
            ]
        avg_ms_by_policy_name = [
            np.median(ms_by_policy_name[p_name]) for p_name, _ in valid_policies
        ]
        self.logger.info(f"Pre-game Analysis: Selected {selected_policy_name}")
        for (p_name, p), avg_ms in zip(valid_policies, avg_ms_by_policy_name):
            self.logger.info(
                f"{p_name}: {self.avg_ms_by_terrain_md5[terrain_md5].get(p_name, 0):.1f} ms "
                f"(updated by {avg_ms:.1f} [{len(ms_by_policy_name[p_name])}])"
            )
        self.selected_policy_for_terrain_md5[terrain_md5] = selected_policy_name
        set_space_transform = getattr(self.env, "set_space_transform")
        if isinstance(selected_policy_name, int):
            set_space_transform(sz=selected_policy_name)
        elif isinstance(selected_policy_name, str):
            set_space_transform(terrain_md5=selected_policy_name)
        else:
            raise ValueError(
                f"Policy name {selected_policy_name} is neither size or terrain_md5"
            )
        return actions_by_policy_name[selected_policy_name]

    def use_policy(
        self,
        policy: Policy,
        policy_name: PolicyName,
        terrain_md5: str,
        obs: VecEnvObs,
        deterministic: bool,
        action_masks: Optional[NumpyOrDict],
    ) -> np.ndarray:
        if self.last_used != policy_name:
            self.logger.info(f"Using policy {policy_name}")
            self.last_used = policy_name
        start_s = time.perf_counter()
        actions = policy.act(
            obs, deterministic=deterministic, action_masks=action_masks
        )
        end_s = time.perf_counter()
        self.avg_ms_by_terrain_md5[terrain_md5].add(
            policy_name, [(end_s - start_s) * 1000]
        )
        return actions
