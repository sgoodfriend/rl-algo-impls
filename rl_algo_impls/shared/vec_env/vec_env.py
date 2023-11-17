import os
from dataclasses import astuple
from typing import Callable, Optional

import gymnasium
from gymnasium.experimental.vector.async_vector_env import AsyncVectorEnv
from gymnasium.experimental.vector.sync_vector_env import SyncVectorEnv
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.resize_observation import ResizeObservation
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, NoopResetEnv
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.runner.config import Config, EnvHyperparams
from rl_algo_impls.shared.policy.policy import VEC_NORMALIZE_FILENAME
from rl_algo_impls.shared.vec_env.utils import (
    import_for_env_id,
    is_atari,
    is_car_racing,
    is_gym_procgen,
    is_lux,
    is_microrts,
)
from rl_algo_impls.wrappers.action_mask_wrapper import SingleActionMaskWrapper
from rl_algo_impls.wrappers.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireOnLifeStarttEnv,
)
from rl_algo_impls.wrappers.episode_stats_writer import EpisodeStatsWriter
from rl_algo_impls.wrappers.hwc_to_chw_observation import HwcToChwObservation
from rl_algo_impls.wrappers.initial_step_truncate_wrapper import (
    InitialStepTruncateWrapper,
)
from rl_algo_impls.wrappers.no_reward_timeout import NoRewardTimeout
from rl_algo_impls.wrappers.noop_env_seed import NoopEnvSeed
from rl_algo_impls.wrappers.normalize import NormalizeObservation, NormalizeReward
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper
from rl_algo_impls.wrappers.vector_env_render_compat import VectorEnvRenderCompat
from rl_algo_impls.wrappers.vector_wrapper import VectorEnv
from rl_algo_impls.wrappers.video_compat_wrapper import VideoCompatWrapper


def make_vec_env(
    config: Config,
    hparams: EnvHyperparams,
    training: bool = True,
    render: bool = False,
    tb_writer: Optional[SummaryWriter] = None,
) -> VectorEnv:
    (
        env_type,
        n_envs,
        frame_stack,
        make_kwargs,
        no_reward_timeout_steps,
        no_reward_fire_steps,
        vec_env_class,
        normalize,
        normalize_kwargs,
        rolling_length,
        video_step_interval,
        initial_steps_to_truncate,
        clip_atari_rewards,
        normalize_type,
        mask_actions,
        _,  # bots
        self_play_kwargs,
        selfplay_bots,
        _,  # additional_win_loss_reward
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

    import_for_env_id(config.env_id)

    seed = config.seed(training=training)

    make_kwargs = make_kwargs.copy() if make_kwargs is not None else {}
    if is_car_racing(config):
        make_kwargs["verbose"] = 0
    if not render:
        make_kwargs["render_mode"] = "rgb_array"

    def make(idx: int) -> Callable[[], gymnasium.Env]:
        def _make() -> gymnasium.Env:
            env = gymnasium.make(config.env_id, **make_kwargs)
            env = RecordEpisodeStatistics(env)
            env = VideoCompatWrapper(env)
            if training and initial_steps_to_truncate:
                env = InitialStepTruncateWrapper(
                    env, idx * initial_steps_to_truncate // n_envs
                )
            if is_atari(config):  # type: ignore
                env = NoopResetEnv(env, noop_max=30)
                env = MaxAndSkipEnv(env, skip=4)
                env = EpisodicLifeEnv(env, training=training)
                action_meanings = env.unwrapped.get_action_meanings()
                if "FIRE" in action_meanings:  # type: ignore
                    env = FireOnLifeStarttEnv(env, action_meanings.index("FIRE"))
                if clip_atari_rewards:
                    env = ClipRewardEnv(env, training=training)
                env = ResizeObservation(env, (84, 84))
                env = GrayScaleObservation(env, keep_dim=False)
                env = FrameStack(env, frame_stack)
            elif is_car_racing(config):
                env = MaxAndSkipEnv(env, skip=2)
                env = ResizeObservation(env, (64, 64))
                env = GrayScaleObservation(env, keep_dim=False)
                env = FrameStack(env, frame_stack)
            elif is_gym_procgen(config):
                # env = GrayScaleObservation(env, keep_dim=False)
                env = NoopEnvSeed(env)
                env = HwcToChwObservation(env)
                if frame_stack > 1:
                    env = FrameStack(env, frame_stack)
            elif is_microrts(config):
                env = HwcToChwObservation(env)

            if no_reward_timeout_steps:
                env = NoRewardTimeout(
                    env, no_reward_timeout_steps, n_fire_steps=no_reward_fire_steps
                )

            if seed is not None:
                env.reset(seed=seed + idx)
                env.action_space.seed(seed + idx)
                env.observation_space.seed(seed + idx)

            return env

        return _make

    if env_type == "gymvec":
        VecEnvClass = {"sync": SyncVectorEnv, "async": AsyncVectorEnv}[vec_env_class]
    else:
        raise ValueError(f"env_type {env_type} unsupported")
    envs = VecEnvClass([make(i) for i in range(n_envs)])
    envs = VectorEnvRenderCompat(envs)
    if mask_actions:
        envs = SingleActionMaskWrapper(envs)

    if self_play_kwargs:
        if selfplay_bots:
            self_play_kwargs["selfplay_bots"] = selfplay_bots
        envs = SelfPlayWrapper(envs, config, **self_play_kwargs)

    if training:
        assert tb_writer
        envs = EpisodeStatsWriter(
            envs, tb_writer, training=training, rolling_length=rolling_length
        )
    if normalize:
        if normalize_type is None:
            normalize_type = "gymlike"
        normalize_kwargs = normalize_kwargs or {}
        if normalize_type == "gymlike":
            if normalize_kwargs.get("norm_obs", True):
                envs = NormalizeObservation(
                    envs, training=training, clip=normalize_kwargs.get("clip_obs", 10.0)
                )
            if training and normalize_kwargs.get("norm_reward", True):
                envs = NormalizeReward(
                    envs,
                    training=training,
                    clip=normalize_kwargs.get("clip_reward", 10.0),
                )
        else:
            raise ValueError(f"normalize_type {normalize_type} not supported (gymlike)")
    return envs
