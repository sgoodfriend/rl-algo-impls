from dataclasses import astuple
from typing import Optional

from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.checkpoints.checkpoints_manager import PolicyCheckpointsManager
from rl_algo_impls.lux.vec_env.vec_lux_env import VecLuxEnv
from rl_algo_impls.lux.vec_env.vec_lux_replay_env import VecLuxReplayEnv
from rl_algo_impls.runner.config import Config, EnvHyperparams
from rl_algo_impls.wrappers.additional_win_loss_reward import (
    AdditionalWinLossRewardWrapper,
)
from rl_algo_impls.wrappers.episode_stats_writer import EpisodeStatsWriter
from rl_algo_impls.wrappers.hwc_to_chw_observation import HwcToChwVectorObservation
from rl_algo_impls.wrappers.info_rewards_wrapper import InfoRewardsWrapper
from rl_algo_impls.wrappers.mask_resettable_episode_statistics import (
    MaskResettableEpisodeStatistics,
)
from rl_algo_impls.wrappers.normalize import NormalizeObservation, NormalizeReward
from rl_algo_impls.wrappers.play_checkpoints_wrapper import PlayCheckpointsWrapper
from rl_algo_impls.wrappers.score_reward_wrapper import ScoreRewardWrapper
from rl_algo_impls.wrappers.self_play_eval_wrapper import SelfPlayEvalWrapper
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper
from rl_algo_impls.wrappers.vector_env_render_compat import VectorEnvRenderCompat
from rl_algo_impls.wrappers.vector_wrapper import VectorEnv


def make_lux_env(
    config: Config,
    hparams: EnvHyperparams,
    training: bool = True,
    render: bool = False,
    tb_writer: Optional[SummaryWriter] = None,
    checkpoints_manager: Optional[PolicyCheckpointsManager] = None,
) -> VectorEnv:
    (
        _,  # env_type,
        n_envs,
        _,  # frame_stack
        make_kwargs,
        _,  # no_reward_timeout_steps
        _,  # no_reward_fire_steps
        vec_env_class,
        normalize,
        normalize_kwargs,
        rolling_length,
        _,  # video_step_interval
        _,  # initial_steps_to_truncate
        _,  # clip_atari_rewards
        normalize_type,
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
        play_checkpoints_kwargs,
        additional_win_loss_smoothing_factor,
        info_rewards,
    ) = astuple(hparams)

    seed = config.seed(training=training)
    make_kwargs = make_kwargs or {}
    self_play_kwargs = self_play_kwargs or {}
    num_envs = (
        n_envs + self_play_kwargs.get("num_old_policies", 0) + len(selfplay_bots or [])
    )
    if play_checkpoints_kwargs:
        assert (
            self_play_kwargs.get("num_old_policies", 0) == 0
        ), "play_checkpoints_kwargs doesn't work with self_play_kwargs"
        assert (
            not selfplay_bots
        ), "play_checkpoints_kwargs doesn't work with selfplay_bots"
        num_envs += play_checkpoints_kwargs["n_envs_against_checkpoints"] or n_envs
    if num_envs == 1 and not training:
        num_envs = 2
    assert num_envs % 2 == 0, f"num_envs {num_envs} must be even"

    if vec_env_class == "sync":
        envs = VecLuxEnv(num_envs, **make_kwargs)
    elif vec_env_class == "replay":
        envs = VecLuxReplayEnv(num_envs, **make_kwargs)
        envs = HwcToChwVectorObservation(envs)
    elif vec_env_class == "ray":
        from rl_algo_impls.lux.vec_env.lux_ray_vector_env import LuxRayVectorEnv

        envs = (
            LuxRayVectorEnv(num_envs, **make_kwargs)
            if num_envs > 2
            else VecLuxEnv(num_envs, **make_kwargs)
        )
    elif vec_env_class == "jux":
        from rl_algo_impls.lux.vec_env.jux_vector_env import JuxVectorEnv

        envs = JuxVectorEnv(num_envs, **make_kwargs)
    else:
        raise ValueError(f"Unknown vec_env_class: {vec_env_class}")
    envs = VectorEnvRenderCompat(envs)

    if play_checkpoints_kwargs:
        assert (
            checkpoints_manager
        ), f"play_checkpoints_kwargs requires checkpoints_manager"
        envs = PlayCheckpointsWrapper(
            envs, checkpoints_manager, **play_checkpoints_kwargs
        )
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
    if info_rewards:
        envs = InfoRewardsWrapper(envs, **info_rewards)
    if normalize:
        if normalize_type is None:
            normalize_type = "gymlike"
        if normalize_type == "gymlike":
            if normalize_kwargs.get("norm_obs", True):
                envs = NormalizeObservation(
                    envs, training=training, clip=normalize_kwargs.get("clip_obs", 10.0)
                )
            if training and normalize_kwargs.get("norm_reward", True):
                rew_shape = (
                    1
                    + (1 if additional_win_loss_reward else 0)
                    + (1 if score_reward_kwargs else 0)
                    + (len(info_rewards["info_paths"]) if info_rewards else 0),
                )
                if rew_shape == (1,):
                    rew_shape = ()
                envs = NormalizeReward(
                    envs,
                    training=training,
                    gamma=normalize_kwargs.get("gamma_reward", 0.99),
                    clip=normalize_kwargs.get("clip_reward", 10.0),
                    shape=rew_shape,
                    exponential_moving_mean_var=normalize_kwargs.get(
                        "exponential_moving_mean_var_reward", False
                    ),
                    emv_window_size=normalize_kwargs.get("emv_window_size", 5e6),
                )
        else:
            raise ValueError(f"normalize_type {normalize_type} not supported (gymlike)")

    return envs
