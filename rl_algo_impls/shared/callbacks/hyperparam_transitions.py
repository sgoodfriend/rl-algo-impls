from typing import Any, Dict, List, Optional

import numpy as np

from rl_algo_impls.ppo.learning_rate_by_kl_divergence import LearningRateByKLDivergence
from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.callbacks.callback import Callback
from rl_algo_impls.shared.data_store.data_store_view import LearnerDataStoreView
from rl_algo_impls.shared.tensor_utils import num_or_array
from rl_algo_impls.utils.interpolate import InterpolateMethod, interpolate

ALGO_SET_NAMES = {
    "multi_reward_weights",
    "vf_coef",
    "learning_rate",
    "clip_range",
    "clip_range_vf",
    "ent_coef",
    "teacher_kl_loss_coef",
}
ALGO_BOOL_NAMES = {"freeze_policy_head", "freeze_value_head", "freeze_backbone"}

ROLLOUT_GENERATOR_NAMES = {
    "gamma",
    "gae_lambda",
    "switch_range",
    "guide_probability",
    "rolling_num_envs_reset_every_rollout",
    "random_num_envs_reset_every_rollout",
}

LEARNING_RATE_BY_KL_DIVERGENCE_NAMES = {
    "target_kl",
}


class HyperparamTransitions(Callback):
    def __init__(
        self,
        config: Config,
        algo: Algorithm,
        learner_data_store_view: LearnerDataStoreView,
        phases: List[Dict[str, Any]],
        durations: List[float],
        start_timesteps: int = 0,
        interpolate_method: str = "linear",
        lr_by_kl_callback: Optional[LearningRateByKLDivergence] = None,
    ) -> None:
        super().__init__()
        self.algo = algo
        self.learner_data_store_view = learner_data_store_view
        self.lr_by_kl_callback = lr_by_kl_callback

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

    def on_step(self, timesteps_elapsed: int = 1, **kwargs) -> bool:
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
            if k in ALGO_SET_NAMES:
                assert hasattr(self.algo, k)
                setattr(self.algo, k, num_or_array(v))
            elif k in ALGO_BOOL_NAMES:
                assert hasattr(self.algo, k)
                setattr(self.algo, k, v)
            elif k in ROLLOUT_GENERATOR_NAMES:
                self.learner_data_store_view.update_rollout_param(k, v)
            elif k in LEARNING_RATE_BY_KL_DIVERGENCE_NAMES:
                assert self.lr_by_kl_callback is not None
                assert hasattr(self.lr_by_kl_callback, k)
                setattr(self.lr_by_kl_callback, k, v)
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
            if k in ALGO_SET_NAMES:
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
            elif k in ROLLOUT_GENERATOR_NAMES:
                self.learner_data_store_view.update_rollout_param(
                    k,
                    interpolate(
                        old_v,
                        next_v,
                        transition_progress,
                        self.interpolate_method,
                    ),
                )
            elif k in LEARNING_RATE_BY_KL_DIVERGENCE_NAMES:
                assert self.lr_by_kl_callback is not None
                assert hasattr(self.lr_by_kl_callback, k)
                setattr(
                    self.lr_by_kl_callback,
                    k,
                    interpolate(
                        old_v,
                        next_v,
                        transition_progress,
                        self.interpolate_method,
                    ),
                )
            else:
                raise ValueError(f"{k} not supported in {self.__class__.__name__}")
