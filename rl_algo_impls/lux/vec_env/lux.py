from dataclasses import astuple
from typing import Callable, Dict, Optional

import gymnasium
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.lux.vec_env.lux_ray_vector_env import LuxRayVectorEnv
from rl_algo_impls.lux.vec_env.vec_lux_env import VecLuxEnv
from rl_algo_impls.lux.vec_env.vec_lux_replay_env import VecLuxReplayEnv
from rl_algo_impls.runner.config import Config, EnvHyperparams
from rl_algo_impls.wrappers.additional_win_loss_reward import (
    AdditionalWinLossRewardWrapper,
)
from rl_algo_impls.wrappers.episode_stats_writer import EpisodeStatsWriter
from rl_algo_impls.wrappers.hwc_to_chw_observation import HwcToChwVectorObservation
from rl_algo_impls.wrappers.mask_resettable_episode_statistics import (
    MaskResettableEpisodeStatistics,
)
from rl_algo_impls.wrappers.score_reward_wrapper import ScoreRewardWrapper
from rl_algo_impls.wrappers.self_play_eval_wrapper import SelfPlayEvalWrapper
from rl_algo_impls.wrappers.self_play_reference_wrapper import SelfPlayReferenceWrapper
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper
from rl_algo_impls.wrappers.vector_env_render_compat import VectorEnvRenderCompat
from rl_algo_impls.wrappers.vector_wrapper import VectorEnv


def make_lux_env(
    config: Config,
    hparams: EnvHyperparams,
    training: bool = True,
    render: bool = False,
    tb_writer: Optional[SummaryWriter] = None,
) -> VectorEnv:
    (
        _,  # env_type,
        n_envs,
        _,  # frame_stack
        make_kwargs,
        _,  # no_reward_timeout_steps
        _,  # no_reward_fire_steps
        vec_env_class,
        _,  # normalize
        _,  # normalize_kwargs,
        rolling_length,
        _,  # video_step_interval
        _,  # initial_steps_to_truncate
        _,  # clip_atari_rewards
        _,  # normalize_type
        _,  # mask_actions
        _,  # bots
        self_play_kwargs,
        selfplay_bots,
        additional_win_loss_reward,
        _,  # map_paths,
        score_reward_kwargs,
        _,  # is_agent
        _,  # valid_sizes,
        _,  # paper_planes_sizes,
        _,  # fixed_size,
        _,  # terrain_overrides,
        _,  # time_budget_ms,
        _,  # video_frames_per_second,
        _,  # reference_bot,
        self_play_reference_kwargs,
        additional_win_loss_smoothing_factor,
    ) = astuple(hparams)

    seed = config.seed(training=training)
    make_kwargs = make_kwargs or {}
    self_play_kwargs = self_play_kwargs or {}
    num_envs = (
        n_envs + self_play_kwargs.get("num_old_policies", 0) + len(selfplay_bots or [])
    )
    if num_envs == 1 and not training:
        # Workaround for supporting the video env
        num_envs = 2

    if vec_env_class == "sync":
        envs = VecLuxEnv(num_envs, **make_kwargs)
    elif vec_env_class == "replay":
        envs = VecLuxReplayEnv(num_envs, **make_kwargs)
        envs = HwcToChwVectorObservation(envs)
    elif vec_env_class == "ray":
        envs = (
            LuxRayVectorEnv(num_envs, **make_kwargs)
            if num_envs > 2
            else VecLuxEnv(num_envs, **make_kwargs)
        )
    else:
        raise ValueError(f"Unknown vec_env_class: {vec_env_class}")
    envs = VectorEnvRenderCompat(envs)

    if self_play_reference_kwargs:
        envs = SelfPlayReferenceWrapper(envs, **self_play_reference_kwargs)
    if self_play_kwargs:
        if not training and self_play_kwargs.get("eval_use_training_cache", False):
            envs = SelfPlayEvalWrapper(envs)
        else:
            if selfplay_bots:
                self_play_kwargs["selfplay_bots"] = selfplay_bots
            envs = SelfPlayWrapper(envs, config, **self_play_kwargs)

    if seed is not None:
        envs.reset(seed=seed)
        envs.action_space.seed(seed)
        envs.observation_space.seed(seed)

    envs = MaskResettableEpisodeStatistics(envs)
    if training and tb_writer:
        envs = EpisodeStatsWriter(
            envs,
            tb_writer,
            training=training,
            rolling_length=rolling_length,
            additional_keys_to_log=config.additional_keys_to_log,
        )

    if additional_win_loss_reward:
        envs = AdditionalWinLossRewardWrapper(
            envs, label_smoothing_factor=additional_win_loss_smoothing_factor
        )
    if score_reward_kwargs:
        envs = ScoreRewardWrapper(envs, **score_reward_kwargs)

    return envs
