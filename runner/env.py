import gym
import numpy as np
import os

from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.frame_stack import FrameStack
from procgen.env import ProcgenEnv
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Callable, Optional, Union

from runner.config import Config, EnvHyperparams
from shared.policy.policy import VEC_NORMALIZE_FILENAME
from wrappers.atari_wrappers import EpisodicLifeEnv, FireOnLifeStarttEnv, ClipRewardEnv
from wrappers.episode_record_video import EpisodeRecordVideo
from wrappers.episode_stats_writer import EpisodeStatsWriter
from wrappers.get_rgb_observation import GetRgbObservation
from wrappers.initial_step_truncate_wrapper import InitialStepTruncateWrapper
from wrappers.is_vector_env import IsVectorEnv
from wrappers.noop_env_seed import NoopEnvSeed
from wrappers.transpose_image_observation import TransposeImageObservation
from wrappers.video_compat_wrapper import VideoCompatWrapper

GeneralVecEnv = Union[VecEnv, gym.vector.VectorEnv, gym.Wrapper]


def make_env(
    config: Config,
    hparams: EnvHyperparams,
    training: bool = True,
    render: bool = False,
    normalize_load_path: Optional[str] = None,
    tb_writer: Optional[SummaryWriter] = None,
) -> GeneralVecEnv:
    if hparams.is_procgen:
        return _make_procgen_env(
            config,
            hparams,
            training=training,
            render=render,
            normalize_load_path=normalize_load_path,
            tb_writer=tb_writer,
        )
    else:
        return _make_vec_env(
            config,
            hparams,
            training=training,
            render=render,
            normalize_load_path=normalize_load_path,
            tb_writer=tb_writer,
        )


def make_eval_env(
    config: Config,
    hparams: EnvHyperparams,
    override_n_envs: Optional[int] = None,
    **kwargs
) -> GeneralVecEnv:
    kwargs = kwargs.copy()
    kwargs["training"] = False
    if override_n_envs is not None:
        hparams_kwargs = hparams._asdict()
        hparams_kwargs["n_envs"] = override_n_envs
        if override_n_envs == 1:
            hparams_kwargs["vec_env_class"] = "dummy"
        hparams = EnvHyperparams(**hparams_kwargs)
    return make_env(config, hparams, **kwargs)


def _make_vec_env(
    config: Config,
    hparams: EnvHyperparams,
    training: bool = True,
    render: bool = False,
    normalize_load_path: Optional[str] = None,
    tb_writer: Optional[SummaryWriter] = None,
) -> GeneralVecEnv:
    (
        _,
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
    ) = hparams

    if "BulletEnv" in config.env_id:
        import pybullet_envs

    spec = gym.spec(config.env_id)
    seed = config.seed(training=training)

    def make(idx: int) -> Callable[[], gym.Env]:
        env_kwargs = make_kwargs.copy() if make_kwargs is not None else {}
        if "BulletEnv" in config.env_id and render:
            env_kwargs["render"] = True
        if "CarRacing" in config.env_id:
            env_kwargs["verbose"] = 0
        if "procgen" in config.env_id:
            if not render:
                env_kwargs["render_mode"] = "rgb_array"

        def _make() -> gym.Env:
            env = gym.make(config.env_id, **env_kwargs)
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

    VecEnvClass = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[vec_env_class]
    venv = VecEnvClass([make(i) for i in range(n_envs)])
    if training:
        assert tb_writer
        venv = EpisodeStatsWriter(
            venv, tb_writer, training=training, rolling_length=rolling_length
        )
    if normalize:
        if normalize_load_path:
            venv = VecNormalize.load(
                os.path.join(normalize_load_path, VEC_NORMALIZE_FILENAME),
                venv,  # type: ignore
            )
        else:
            venv = VecNormalize(
                venv,  # type: ignore
                training=training,
                **(normalize_kwargs or {}),
            )
        if not training:
            venv.norm_reward = False
    return venv


def _make_procgen_env(
    config: Config,
    hparams: EnvHyperparams,
    training: bool = True,
    render: bool = False,
    normalize_load_path: Optional[str] = None,
    tb_writer: Optional[SummaryWriter] = None,
) -> GeneralVecEnv:
    (
        _,
        n_envs,
        frame_stack,
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
    ) = hparams

    seed = config.seed(training=training)

    make_kwargs = make_kwargs or {}
    if not render:
        make_kwargs["render_mode"] = "rgb_array"
    if seed is not None:
        make_kwargs["rand_seed"] = seed

    envs = ProcgenEnv(n_envs, config.env_id, **make_kwargs)
    envs = IsVectorEnv(envs)
    envs = GetRgbObservation(envs)
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
    if normalize:
        normalize_kwargs = normalize_kwargs or {}
        # TODO: Handle reward stats saving/loading/syncing, but it's only important
        # for checkpointing
        envs = gym.wrappers.NormalizeReward(envs)
        clip_obs = normalize_kwargs.get("clip_reward", 10.0)
        envs = gym.wrappers.TransformReward(
            envs, lambda r: np.clip(r, -clip_obs, clip_obs)
        )

    return envs
