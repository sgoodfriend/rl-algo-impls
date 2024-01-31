from dataclasses import astuple
from typing import Optional

from gymnasium.experimental.wrappers.vector.record_episode_statistics import (
    RecordEpisodeStatisticsV0,
)

from rl_algo_impls.runner.config import Config
from rl_algo_impls.runner.env_hyperparams import EnvHyperparams
from rl_algo_impls.shared.data_store.data_store_view import VectorEnvDataStoreView
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.wrappers.episode_stats_writer import EpisodeStatsWriter
from rl_algo_impls.wrappers.hwc_to_chw_observation import HwcToChwVectorObservation
from rl_algo_impls.wrappers.is_vector_env import IsVectorEnv
from rl_algo_impls.wrappers.normalize import NormalizeReward
from rl_algo_impls.wrappers.vector_wrapper import VectorEnv


def make_procgen_env(
    config: Config,
    hparams: EnvHyperparams,
    data_store_view: VectorEnvDataStoreView,
    training: bool = True,
    render: bool = False,
    tb_writer: Optional[AbstractSummaryWrapper] = None,
    **kwargs,
) -> VectorEnv:
    from gym3 import ExtractDictObWrapper, ViewerWrapper
    from procgen.env import ProcgenGym3Env, ToBaselinesVecEnv

    (
        _,  # env_type
        n_envs,
        _,  # frame_stack
        make_kwargs,
        _,  # no_reward_timeout_steps
        _,  # no_reward_fire_steps
        _,  # vec_env_class
        normalize,
        normalize_kwargs,
        rolling_length,
        _,  # train_record_video
        _,  # video_step_interval
        _,  # initial_steps_to_truncate
        _,  # clip_atari_rewards
        _,  # normalize_type
        _,  # mask_actions
        _,  # bots
        _,  # self_play_kwargs
        _,  # selfplay_bots
        _,  # additional_win_loss_reward,
        _,  # map_paths,
        _,  # score_reward_kwargs,
        _,  # is_agent
        _,  # valid_sizes,
        _,  # paper_planes_sizes,
        _,  # fixed_size,
        _,  # terrain_overrides,
        _,  # time_budget_ms,
        _,  # video_frames_per_second,
        _,  # reference_bot,
        _,  # play_checkpoints_kwargs,
        _,  # additional_win_loss_smoothing_factor,
        _,  # info_rewards,
    ) = astuple(hparams)

    seed = config.seed(training=training)

    make_kwargs = make_kwargs or {}
    make_kwargs["render_mode"] = "rgb_array"
    if seed is not None:
        make_kwargs["rand_seed"] = seed

    envs = ProcgenGym3Env(n_envs, config.env_id, **make_kwargs)
    envs = ExtractDictObWrapper(envs, key="rgb")
    if render:
        envs = ViewerWrapper(envs, info_key="rgb")
    envs = ToBaselinesVecEnv(envs)
    envs = IsVectorEnv(envs)
    # TODO: Handle Grayscale and/or FrameStack
    envs = HwcToChwVectorObservation(envs)

    envs = RecordEpisodeStatisticsV0(envs)

    if seed is not None:
        envs.action_space.seed(seed)
        envs.observation_space.seed(seed)

    if training:
        assert tb_writer
        envs = EpisodeStatsWriter(
            envs, tb_writer, training=training, rolling_length=rolling_length
        )
    if normalize and training:
        envs = NormalizeReward(
            envs,
            data_store_view,
            training=training,
            clip=normalize_kwargs.get("clip_reward", 10.0),
        )

    return envs  # type: ignore
