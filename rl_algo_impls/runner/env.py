import gym
import numpy as np
import os

from dataclasses import asdict, astuple
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.frame_stack import FrameStack
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Callable, Optional

from rl_algo_impls.runner.config import Config, EnvHyperparams
from rl_algo_impls.shared.policy.policy import VEC_NORMALIZE_FILENAME
from rl_algo_impls.wrappers.atari_wrappers import (
    EpisodicLifeEnv,
    FireOnLifeStarttEnv,
    ClipRewardEnv,
)
from rl_algo_impls.wrappers.episode_record_video import EpisodeRecordVideo
from rl_algo_impls.wrappers.episode_stats_writer import EpisodeStatsWriter
from rl_algo_impls.wrappers.initial_step_truncate_wrapper import (
    InitialStepTruncateWrapper,
)
from rl_algo_impls.wrappers.is_vector_env import IsVectorEnv
from rl_algo_impls.wrappers.noop_env_seed import NoopEnvSeed
from rl_algo_impls.wrappers.normalize import NormalizeObservation, NormalizeReward
from rl_algo_impls.wrappers.sync_vector_env_render_compat import (
    SyncVectorEnvRenderCompat,
)
from rl_algo_impls.wrappers.transpose_image_observation import TransposeImageObservation
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv
from rl_algo_impls.wrappers.video_compat_wrapper import VideoCompatWrapper


def make_env(
    config: Config,
    hparams: EnvHyperparams,
    training: bool = True,
    render: bool = False,
    normalize_load_path: Optional[str] = None,
    tb_writer: Optional[SummaryWriter] = None,
) -> VecEnv:
    if hparams.env_type == "procgen":
        return _make_procgen_env(
            config,
            hparams,
            training=training,
            render=render,
            normalize_load_path=normalize_load_path,
            tb_writer=tb_writer,
        )
    elif hparams.env_type in {"sb3vec", "gymvec"}:
        return _make_vec_env(
            config,
            hparams,
            training=training,
            render=render,
            normalize_load_path=normalize_load_path,
            tb_writer=tb_writer,
        )
    else:
        raise ValueError(f"env_type {hparams.env_type} not supported")


def make_eval_env(
    config: Config,
    hparams: EnvHyperparams,
    override_n_envs: Optional[int] = None,
    **kwargs,
) -> VecEnv:
    kwargs = kwargs.copy()
    kwargs["training"] = False
    if override_n_envs is not None:
        hparams_kwargs = asdict(hparams)
        hparams_kwargs["n_envs"] = override_n_envs
        if override_n_envs == 1:
            hparams_kwargs["vec_env_class"] = "sync"
        hparams = EnvHyperparams(**hparams_kwargs)
    return make_env(config, hparams, **kwargs)


def _make_vec_env(
    config: Config,
    hparams: EnvHyperparams,
    training: bool = True,
    render: bool = False,
    normalize_load_path: Optional[str] = None,
    tb_writer: Optional[SummaryWriter] = None,
) -> VecEnv:
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
        train_record_video,
        video_step_interval,
        initial_steps_to_truncate,
        clip_atari_rewards,
    ) = astuple(hparams)

    if "BulletEnv" in config.env_id:
        import pybullet_envs

    spec = gym.spec(config.env_id)
    seed = config.seed(training=training)

    make_kwargs = make_kwargs.copy() if make_kwargs is not None else {}
    if "BulletEnv" in config.env_id and render:
        make_kwargs["render"] = True
    if "CarRacing" in config.env_id:
        make_kwargs["verbose"] = 0
    if "procgen" in config.env_id:
        if not render:
            make_kwargs["render_mode"] = "rgb_array"

    def make(idx: int) -> Callable[[], gym.Env]:
        def _make() -> gym.Env:
            env = gym.make(config.env_id, **make_kwargs)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = VideoCompatWrapper(env)
            if training and train_record_video and idx == 0:
                env = EpisodeRecordVideo(
                    env,
                    config.video_prefix,
                    step_increment=n_envs,
                    video_step_interval=int(video_step_interval),
                )
            if training and initial_steps_to_truncate:
                env = InitialStepTruncateWrapper(
                    env, idx * initial_steps_to_truncate // n_envs
                )
            if "AtariEnv" in spec.entry_point:  # type: ignore
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
            elif "CarRacing" in config.env_id:
                env = ResizeObservation(env, (64, 64))
                env = GrayScaleObservation(env, keep_dim=False)
                env = FrameStack(env, frame_stack)
            elif "procgen" in config.env_id:
                # env = GrayScaleObservation(env, keep_dim=False)
                env = NoopEnvSeed(env)
                env = TransposeImageObservation(env)
                if frame_stack > 1:
                    env = FrameStack(env, frame_stack)

            if no_reward_timeout_steps:
                from wrappers.no_reward_timeout import NoRewardTimeout

                env = NoRewardTimeout(
                    env, no_reward_timeout_steps, n_fire_steps=no_reward_fire_steps
                )

            if seed is not None:
                env.seed(seed + idx)
                env.action_space.seed(seed + idx)
                env.observation_space.seed(seed + idx)

            return env

        return _make

    if env_type == "sb3vec":
        VecEnvClass = {"sync": DummyVecEnv, "async": SubprocVecEnv}[vec_env_class]
    elif env_type == "gymvec":
        VecEnvClass = {"sync": SyncVectorEnv, "async": AsyncVectorEnv}[vec_env_class]
    else:
        raise ValueError(f"env_type {env_type} unsupported")
    envs = VecEnvClass([make(i) for i in range(n_envs)])
    if env_type == "gymvec" and vec_env_class == "sync":
        envs = SyncVectorEnvRenderCompat(envs)
    if training:
        assert tb_writer
        envs = EpisodeStatsWriter(
            envs, tb_writer, training=training, rolling_length=rolling_length
        )
    if normalize:
        normalize_kwargs = normalize_kwargs or {}
        if env_type == "sb3vec":
            if normalize_load_path:
                envs = VecNormalize.load(
                    os.path.join(normalize_load_path, VEC_NORMALIZE_FILENAME),
                    envs,  # type: ignore
                )
            else:
                envs = VecNormalize(
                    envs,  # type: ignore
                    training=training,
                    **normalize_kwargs,
                )
            if not training:
                envs.norm_reward = False
        else:
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
    return envs


def _make_procgen_env(
    config: Config,
    hparams: EnvHyperparams,
    training: bool = True,
    render: bool = False,
    normalize_load_path: Optional[str] = None,
    tb_writer: Optional[SummaryWriter] = None,
) -> VecEnv:
    from gym3 import ViewerWrapper, ExtractDictObWrapper
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
    envs = TransposeImageObservation(envs)

    envs = gym.wrappers.RecordEpisodeStatistics(envs)

    if seed is not None:
        envs.action_space.seed(seed)
        envs.observation_space.seed(seed)

    if training:
        assert tb_writer
        envs = EpisodeStatsWriter(
            envs, tb_writer, training=training, rolling_length=rolling_length
        )
    if normalize and training:
        normalize_kwargs = normalize_kwargs or {}
        envs = gym.wrappers.NormalizeReward(envs)
        clip_obs = normalize_kwargs.get("clip_reward", 10.0)
        envs = gym.wrappers.TransformReward(
            envs, lambda r: np.clip(r, -clip_obs, clip_obs)
        )

    return envs  # type: ignore
