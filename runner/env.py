import gym
import os

from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.frame_stack import FrameStack
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Any, Callable, Dict, Optional, Union

from runner.names import Names
from shared.policy.policy import VEC_NORMALIZE_FILENAME
from wrappers.atari_wrappers import EpisodicLifeEnv, FireOnLifeStarttEnv, ClipRewardEnv
from wrappers.episode_record_video import EpisodeRecordVideo
from wrappers.episode_stats_writer import EpisodeStatsWriter
from wrappers.initial_step_offset_wrapper import InitialStepOffsetWrapper
from wrappers.video_compat_wrapper import VideoCompatWrapper


def make_env(
    names: Names,
    training: bool = True,
    render: bool = False,
    normalize_load_path: Optional[str] = None,
    n_envs: int = 1,
    frame_stack: int = 1,
    make_kwargs: Optional[Dict[str, Any]] = None,
    no_reward_timeout_steps: Optional[int] = None,
    no_reward_fire_steps: Optional[int] = None,
    vec_env_class: str = "dummy",
    normalize: bool = False,
    normalize_kwargs: Optional[Dict[str, Any]] = None,
    tb_writer: Optional[SummaryWriter] = None,
    rolling_length: int = 100,
    train_record_video: bool = False,
    video_step_interval: Union[int, float] = 1_000_000,
    initialize_steps_to_offset: Optional[int] = None,
) -> VecEnv:
    if "BulletEnv" in names.env_id:
        import pybullet_envs

    make_kwargs = make_kwargs if make_kwargs is not None else {}
    if "BulletEnv" in names.env_id and render:
        make_kwargs["render"] = True
    if "CarRacing" in names.env_id:
        make_kwargs["verbose"] = 0

    spec = gym.spec(names.env_id)

    def make(idx: int) -> Callable[[], gym.Env]:
        def _make() -> gym.Env:
            env = gym.make(names.env_id, **make_kwargs)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = VideoCompatWrapper(env)
            if training and train_record_video and idx == 0:
                env = EpisodeRecordVideo(
                    env,
                    names.video_prefix,
                    step_increment=n_envs,
                    video_step_interval=int(video_step_interval),
                )
            if training and initialize_steps_to_offset:
                env = InitialStepOffsetWrapper(env, idx * initialize_steps_to_offset // n_envs)
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
            elif "CarRacing" in names.env_id:
                env = ResizeObservation(env, (64, 64))
                env = GrayScaleObservation(env, keep_dim=False)
                env = FrameStack(env, frame_stack)

            if no_reward_timeout_steps:
                from wrappers.no_reward_timeout import NoRewardTimeout

                env = NoRewardTimeout(
                    env, no_reward_timeout_steps, n_fire_steps=no_reward_fire_steps
                )

            seed = names.seed(training=training)
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
                os.path.join(normalize_load_path, VEC_NORMALIZE_FILENAME), venv
            )
        else:
            venv = VecNormalize(venv, training=training, **(normalize_kwargs or {}))
        if not training:
            venv.norm_reward = False
    return venv


def make_eval_env(
    names: Names, override_n_envs: Optional[int] = None, **kwargs
) -> VecEnv:
    kwargs = kwargs.copy()
    kwargs["training"] = False
    if override_n_envs is not None:
        kwargs["n_envs"] = override_n_envs
        if override_n_envs == 1:
            kwargs["vec_env_class"] = "dummy"
    return make_env(names, **kwargs)
