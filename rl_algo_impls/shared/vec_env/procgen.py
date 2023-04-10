from dataclasses import astuple
from typing import Optional

import gym
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.runner.config import Config, EnvHyperparams
from rl_algo_impls.wrappers.episode_stats_writer import EpisodeStatsWriter
from rl_algo_impls.wrappers.hwc_to_chw_observation import HwcToChwObservation
from rl_algo_impls.wrappers.is_vector_env import IsVectorEnv
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv


def make_procgen_env(
    config: Config,
    hparams: EnvHyperparams,
    training: bool = True,
    render: bool = False,
    normalize_load_path: Optional[str] = None,
    tb_writer: Optional[SummaryWriter] = None,
) -> VecEnv:
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
    envs = HwcToChwObservation(envs)

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
