import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    DefaultDict,
    Dict,
    Generic,
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
from rl_algo_impls.runner.running_utils import load_hyperparams, make_eval_policy
from rl_algo_impls.shared.data_store.in_process_data_store_accessor import (
    InProcessDataStoreAccessor,
)
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.tensor_utils import NumpyOrDict
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces
from rl_algo_impls.wrappers.vector_wrapper import ObsType, VectorEnv

MODEL_ROOT_PATH = "rai_microrts_saved_models"
EXPONENTIAL_MOVING_AVERAGE_SPAN = 100
PRE_GAME_ANALYSIS_BUFFER_MILLISECONDS = 200
TIME_BUDGET_FACTOR = 0.75

file_path = os.path.abspath(Path(__file__))
root_dir = str(Path(file_path).parent.parent.parent.absolute())


@dataclass
class PickerArgs(RunArgs):
    best: bool = True
    use_paper_obs: bool = False
    size: Optional[int] = None
    map_name: Optional[str] = None
    is_final_valid_model: bool = False

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {self.average:.1f} ({self.span})"


T = TypeVar("T")


class ExponentialMovingAverages:
    ema_by_key: Dict[str, ExponentialMovingAverage]
    span: int

    def __init__(
        self, span: int, initial_ema_by_key: Optional[Dict[str, float]] = None
    ) -> None:
        self.ema_by_key = {}
        self.span = span
        if initial_ema_by_key:
            for key, initial_ema in initial_ema_by_key.items():
                self.ema_by_key[key] = ExponentialMovingAverage([initial_ema], span)

    def add(self, key: str, values: Sequence[float]) -> float:
        if key not in self.ema_by_key:
            self.ema_by_key[key] = ExponentialMovingAverage(values, self.span)
        else:
            self.ema_by_key[key].add_values(values)
        return self.ema_by_key[key].average

    def get(self, key: str, default_value: T = None) -> Union[float, T]:
        if key in self.ema_by_key:
            return self.ema_by_key[key].average
        else:
            return default_value

    def to_dict(self) -> Dict[str, float]:
        return {key: ema.average for key, ema in self.ema_by_key.items()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {json.dumps(self.ema_by_key)}"


class MapSizePolicyPicker(Policy, Generic[ObsType]):
    logger = logging.getLogger("MapSizePolicyPicker")
    avg_ms_by_terrain_md5: DefaultDict[str, ExponentialMovingAverages]
    selected_policy_for_terrain_md5: Dict[str, str]
    last_used: Optional[str] = None

    def __init__(
        self,
        picker_args_by_size: Dict[int, List[PickerArgs]],
        picker_args_by_terrain_md5: Dict[str, List[PickerArgs]],
        env: VectorEnv,
        device: torch.device,
        envs_by_name: Dict[str, VectorEnv],
        time_budget_ms: int,
        use_best_models: bool,
    ) -> None:
        super().__init__(env)
        self.to(device)
        self.envs_by_name = envs_by_name
        self.time_budget_ms = time_budget_ms
        self.use_best_models = use_best_models

        self.avg_ms_by_terrain_md5 = defaultdict(
            lambda: ExponentialMovingAverages(EXPONENTIAL_MOVING_AVERAGE_SPAN)
        )
        self.selected_policy_for_terrain_md5 = {}

        def load_policy(args: PickerArgs, vec_env: VectorEnv) -> Policy:
            policy_hyperparams = args.config.policy_hyperparams
            if "load_run_path" in policy_hyperparams:
                del policy_hyperparams["load_run_path"]
            if "load_run_path_best" in policy_hyperparams:
                del policy_hyperparams["load_run_path_best"]
            if "load_path" in policy_hyperparams:
                del policy_hyperparams["load_path"]
            self.logger.debug(f"Loading policy {args.env}")
            return make_eval_policy(
                args.config,
                EnvSpaces.from_vec_env(vec_env),
                device,
                InProcessDataStoreAccessor(),
                load_path=args.model_path,
                **policy_hyperparams,
            )

        self.args_by_name: Dict[str, PickerArgs] = {}
        policies_by_name: Dict[str, Policy] = {}
        self.policy_names_by_size: Dict[int, List[str]] = {}
        for sz, picker_args in picker_args_by_size.items():
            self.policy_names_by_size[sz] = []
            for p_arg in picker_args:
                p_name = p_arg.env
                self.args_by_name[p_name] = p_arg
                self.policy_names_by_size[sz].append(p_name)
                policies_by_name[p_name] = load_policy(p_arg, envs_by_name[p_name])
        self.policy_names_by_terrain_md5 = {}
        for terrain_md5, picker_args in picker_args_by_terrain_md5.items():
            self.policy_names_by_terrain_md5[terrain_md5] = []
            for p_arg in picker_args:
                p_name = p_arg.env
                self.args_by_name[p_name] = p_arg
                self.policy_names_by_terrain_md5[terrain_md5].append(p_name)
                policies_by_name[p_name] = load_policy(p_arg, envs_by_name[p_name])
        self.policies_by_name = nn.ModuleDict(policies_by_name)

    def terrain_md5(self) -> str:
        get_terrain_md5 = getattr(self.env, "terrain_md5")
        terrain_md5s = get_terrain_md5()
        return terrain_md5s[0]

    def get_policy_by_name(self, name: str) -> Policy:
        p = self.policies_by_name[name]
        assert isinstance(p, Policy)
        return p

    def _valid_policies(self, obs: ObsType) -> List[Tuple[str, Policy]]:
        global errored_on_sizes

        valid_policies: List[Tuple[str, Policy]] = []

        # Returns whether to continue adding policies.
        def add_policies(policy_names: List[str]) -> bool:
            for p_name in policy_names:
                valid_policies.append((p_name, self.get_policy_by_name(p_name)))
                if (
                    self.use_best_models
                    or self.args_by_name[p_name].is_final_valid_model
                ):
                    return False
            return True

        terrain_md5 = self.terrain_md5()
        if not add_policies(self.policy_names_by_terrain_md5.get(terrain_md5, [])):
            return valid_policies

        assert isinstance(obs, np.ndarray)
        obs_size = obs.shape[-1]
        for sz in self.policy_names_by_size:
            if sz >= obs_size:
                if not add_policies(self.policy_names_by_size[sz]):
                    return valid_policies
                break
        else:
            sz = max(self.policy_names_by_size)
            if obs_size not in errored_on_sizes:
                self.logger.error(
                    f"Obseration size {obs_size} exceeded {sz}, using biggest model"
                )
                errored_on_sizes.add(obs_size)
            if not add_policies(self.policy_names_by_size[sz]):
                return valid_policies

        assert valid_policies, f"Found no valid policies for map size {obs_size}"
        return valid_policies

    def act(
        self,
        obs: ObsType,
        deterministic: bool = True,
        action_masks: Optional[NumpyOrDict] = None,
    ) -> np.ndarray:
        terrain_md5 = self.terrain_md5()
        if terrain_md5 in self.selected_policy_for_terrain_md5:
            policy_name = self.selected_policy_for_terrain_md5[terrain_md5]
            policy = self.get_policy_by_name(policy_name)
        else:
            valid_policies = self._valid_policies(obs)
            policy_name, policy = valid_policies[0]
        return self.use_policy(
            policy, policy_name, terrain_md5, obs, deterministic, action_masks
        )

    def pre_game_analysis(
        self,
        obs: ObsType,
        deterministic: bool = True,
        action_masks: Optional[NumpyOrDict] = None,
    ) -> np.ndarray:
        terrain_md5 = self.terrain_md5()
        pre_game_analysis_folder = getattr(self.env, "pre_game_analysis_folder")
        pre_game_analysis_filepath = None
        if pre_game_analysis_folder:
            pre_game_analysis_filepath = os.path.join(
                pre_game_analysis_folder, "avg_ms_by_terrain_policy.json"
            )
            if os.path.exists(pre_game_analysis_filepath):
                with open(pre_game_analysis_filepath) as f:
                    avg_ms_by_terrain_policy = json.loads(f.read())
                    if (
                        terrain_md5 in avg_ms_by_terrain_policy
                        and terrain_md5 not in self.avg_ms_by_terrain_md5
                    ):
                        self.logger.info(f"Loading EMA for {terrain_md5}")
                        self.avg_ms_by_terrain_md5[
                            terrain_md5
                        ] = ExponentialMovingAverages(
                            EXPONENTIAL_MOVING_AVERAGE_SPAN,
                            avg_ms_by_terrain_policy[terrain_md5],
                        )

        valid_policies = self._valid_policies(obs)
        pga_expiration = getattr(self.env, "pre_game_analysis_expiration_ms")
        observation_by_policy_name: Dict[
            str, Tuple[ObsType, Optional[NumpyOrDict]]
        ] = {}
        for p_name, p in valid_policies:
            p_env = self.envs_by_name[p_name]
            observation_by_policy_name[p_name] = (
                p_env.reset()[0],
                getattr(p_env, "get_action_mask")(),
            )

        idx = 0
        ms_by_policy_name: DefaultDict[str, List[float]] = defaultdict(list)
        start_ms = time.perf_counter() * 1000
        actions_by_policy_name: Dict[str, np.ndarray] = {}
        is_first_analysis_for_map = pga_expiration - start_ms > 1000
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
            if self.avg_ms_by_terrain_md5[terrain_md5].get(p_name, 0) < (
                self.time_budget_ms * TIME_BUDGET_FACTOR
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
        self.selected_policy_for_terrain_md5[terrain_md5] = selected_policy_name

        set_space_transform = getattr(self.env, "set_space_transform")
        picker_args = self.args_by_name[selected_policy_name]
        set_space_transform(
            picker_args.size,
            picker_args.use_paper_obs,
            int(self.avg_ms_by_terrain_md5[terrain_md5].get(selected_policy_name, 0)),
        )

        self.logger.info(f"Pre-game Analysis: Selected {selected_policy_name}")
        for p_name, p in valid_policies:
            avg_ms = np.mean(ms_by_policy_name[p_name])
            num_runs = len(ms_by_policy_name[p_name])
            self.logger.info(
                f"{p_name}: {self.avg_ms_by_terrain_md5[terrain_md5].get(p_name, 0):.1f} ms "
                f"(updated by {avg_ms:.1f} [{num_runs}])"
            )

        if is_first_analysis_for_map and pre_game_analysis_filepath:
            with open(pre_game_analysis_filepath, "w") as f:
                f.write(
                    json.dumps(
                        {
                            t_md5: ema.to_dict()
                            for t_md5, ema in self.avg_ms_by_terrain_md5.items()
                        }
                    )
                )

        return actions_by_policy_name[selected_policy_name]

    def use_policy(
        self,
        policy: Policy,
        policy_name: str,
        terrain_md5: str,
        obs: ObsType,
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
