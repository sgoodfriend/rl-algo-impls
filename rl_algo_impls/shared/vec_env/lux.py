from dataclasses import astuple
from typing import Optional

import gym
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.runner.config import Config, EnvHyperparams
from rl_algo_impls.shared.vec_env.vec_lux_env import VecLuxEnv
from rl_algo_impls.wrappers.episode_stats_writer import EpisodeStatsWriter
from rl_algo_impls.wrappers.hwc_to_chw_observation import HwcToChwObservation
from rl_algo_impls.wrappers.self_play_eval_wrapper import SelfPlayEvalWrapper
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv


def make_lux_env(
    config: Config,
    hparams: EnvHyperparams,
    training: bool = True,
    render: bool = False,
    normalize_load_path: Optional[str] = None,
    tb_writer: Optional[SummaryWriter] = None,
) -> VecEnv:
    (
        _,  # env_type
        n_envs,
        _,  # frame_stack
        make_kwargs,
        _,  # no_reward_timeout_steps
        _,  # no_reward_fire_steps
        _,  # vec_env_class
        _,  # normalize
        _,  # normalize_kwargs,
        rolling_length,
        _,  # train_record_video
        _,  # video_step_interval
        _,  # initial_steps_to_truncate
        _,  # clip_atari_rewards
        _,  # normalize_type
        _,  # mask_actions
        _,  # bots
        self_play_kwargs,
        selfplay_bots,
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
    envs = VecLuxEnv(num_envs, **make_kwargs)
    envs = HwcToChwObservation(envs)
    if self_play_kwargs:
        if not training and self_play_kwargs.get("eval_use_training_cache", False):
            envs = SelfPlayEvalWrapper(envs)
        else:
            if selfplay_bots:
                self_play_kwargs["selfplay_bots"] = selfplay_bots
            envs = SelfPlayWrapper(envs, config, **self_play_kwargs)

    if seed is not None:
        envs.action_space.seed(seed)
        envs.observation_space.seed(seed)

    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    if training:
        assert tb_writer
        envs = EpisodeStatsWriter(
            envs,
            tb_writer,
            training=training,
            rolling_length=rolling_length,
            additional_keys_to_log=config.additional_keys_to_log,
        )

    return envs
