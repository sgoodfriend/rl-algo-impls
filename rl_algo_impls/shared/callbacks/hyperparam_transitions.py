from typing import Any, Dict, List, Optional

import numpy as np

from rl_algo_impls.lux.rewards import LuxRewardWeights
from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.callbacks.callback import Callback
from rl_algo_impls.shared.schedule import constant_schedule
from rl_algo_impls.shared.tensor_utils import num_or_array
from rl_algo_impls.utils.interpolate import InterpolateMethod, interpolate
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv

ALGO_SET_NAMES = {"gae_lambda", "multi_reward_weights", "vf_coef"}
ALGO_SET_SCHEDULE_NAMES = {"gamma", "ent_coef", "learning_rate"}
ALGO_BOOL_NAMES = {"freeze_policy_head", "freeze_value_head", "freeze_backbone"}

LUX_REWARD_WEIGHTS_NAME = "reward_weights"


class HyperparamTransitions(Callback):
    def __init__(
        self,
        config: Config,
        env: VecEnv,
        algo: Algorithm,
        phases: List[Dict[str, Any]],
        durations: List[float],
        start_timesteps: int = 0,
        interpolate_method: str = "linear",
    ) -> None:
        super().__init__()
        self.env = env
        self.algo = algo

        self.phases = phases
        assert (
            len(durations) == len(phases) * 2 - 1
        ), f"Durations expected to be 2*len(phases)-1 to account for transitions between phases"
        assert np.isclose(np.sum(durations), 1)
        self.durations = durations

        self.total_train_timesteps = config.n_timesteps
        self.timesteps_elapsed = start_timesteps
        self.interpolate_method = InterpolateMethod[interpolate_method.upper()]
        self.current_phase_idx: Optional[int] = None

        self.update()

    def on_step(self, timesteps_elapsed: int = 1) -> bool:
        super().on_step(timesteps_elapsed)
        self.update()
        return True

    def update(self) -> None:
        progress = self.timesteps_elapsed / self.total_train_timesteps
        prior_duration_accumulation = 0
        current_duration_accumulation = 0
        for idx, d in enumerate(self.durations):
            current_duration_accumulation += d
            if progress < current_duration_accumulation:
                current_or_prior_phase = idx // 2
                if idx % 2 == 0:
                    self.maybe_update_phase(current_or_prior_phase)
                else:
                    self.update_phase_transition(
                        current_or_prior_phase,
                        (progress - prior_duration_accumulation) / d,
                    )
                break
            prior_duration_accumulation = current_duration_accumulation
        else:
            self.maybe_update_phase(len(self.phases) - 1)

    def maybe_update_phase(self, phase_idx: int) -> None:
        if phase_idx == self.current_phase_idx:
            return
        self.current_phase_idx = phase_idx

        phase = self.phases[phase_idx]
        print(f"{self.timesteps_elapsed}: Entering phase {phase_idx}: {phase}")
        for k, v in phase.items():
            if k in ALGO_SET_SCHEDULE_NAMES:
                name = f"{k}_schedule"
                assert hasattr(self.algo, name)
                setattr(self.algo, name, constant_schedule(num_or_array(v)))
            elif k in ALGO_SET_NAMES:
                assert hasattr(self.algo, k)
                setattr(self.algo, k, num_or_array(v))
            elif k in ALGO_BOOL_NAMES:
                assert hasattr(self.algo, k)
                setattr(self.algo, k, v)
            elif k == LUX_REWARD_WEIGHTS_NAME:
                assert hasattr(self.env, k)
                setattr(self.env.unwrapped, k, LuxRewardWeights(**v))
            else:
                raise ValueError(f"{k} not supported in {self.__class__.__name__}")

    def update_phase_transition(
        self, prior_phase_idx: int, transition_progress: float
    ) -> None:
        if self.current_phase_idx is not None:
            print(f"{self.timesteps_elapsed}: Exiting phase {self.current_phase_idx}")
        self.current_phase_idx = None
        prior_phase = self.phases[prior_phase_idx]
        next_phase = self.phases[prior_phase_idx + 1]
        assert set(prior_phase.keys()) == set(
            next_phase.keys()
        ), f"An override has to be specified in every phase"
        for k, next_v in next_phase.items():
            old_v = prior_phase[k]
            if k in ALGO_SET_SCHEDULE_NAMES:
                name = f"{k}_schedule"
                assert hasattr(self.algo, name)
                setattr(
                    self.algo,
                    name,
                    constant_schedule(
                        interpolate(
                            num_or_array(old_v),
                            num_or_array(next_v),
                            transition_progress,
                            self.interpolate_method,
                        )
                    ),
                )
            elif k in ALGO_SET_NAMES:
                assert hasattr(self.algo, k)
                setattr(
                    self.algo,
                    k,
                    interpolate(
                        num_or_array(old_v),
                        num_or_array(next_v),
                        transition_progress,
                        self.interpolate_method,
                    ),
                )
            elif k in ALGO_BOOL_NAMES:
                assert hasattr(self.algo, k)
                setattr(self.algo, k, old_v)
            elif k == LUX_REWARD_WEIGHTS_NAME:
                assert hasattr(self.env, k)
                setattr(
                    self.env.unwrapped,
                    k,
                    LuxRewardWeights.interpolate(
                        old_v, next_v, transition_progress, self.interpolate_method
                    ),
                )
            else:
                raise ValueError(f"{k} not supported in {self.__class__.__name__}")
