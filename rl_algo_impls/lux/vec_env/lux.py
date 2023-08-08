from dataclasses import astuple
from typing import Callable, Dict, Optional

import gym
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.lux.vec_env.lux_async_vector_env import LuxAsyncVectorEnv
from rl_algo_impls.lux.vec_env.vec_lux_env import VecLuxEnv
from rl_algo_impls.lux.vec_env.vec_lux_replay_env import VecLuxReplayEnv
from rl_algo_impls.lux.wrappers.lux_env_gridnet import LuxEnvGridnet
from rl_algo_impls.runner.config import Config, EnvHyperparams
from rl_algo_impls.wrappers.additional_win_loss_reward import (
    AdditionalWinLossRewardWrapper,
)
from rl_algo_impls.wrappers.episode_stats_writer import EpisodeStatsWriter
from rl_algo_impls.wrappers.hwc_to_chw_observation import HwcToChwObservation
from rl_algo_impls.wrappers.score_reward_wrapper import ScoreRewardWrapper
from rl_algo_impls.wrappers.self_play_eval_wrapper import SelfPlayEvalWrapper
from rl_algo_impls.wrappers.self_play_reference_wrapper import SelfPlayReferenceWrapper
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
        _,  # train_record_video
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

    def make(idx: int) -> Callable[[], gym.Env]:
        def _make() -> gym.Env:
            def _gridnet(
                bid_std_dev=5,
                reward_weights: Optional[Dict[str, float]] = None,
                verify: bool = False,
                **kwargs,
            ) -> LuxEnvGridnet:
                return LuxEnvGridnet(
                    gym.make("LuxAI_S2-v0", collect_stats=True, **kwargs),
                    bid_std_dev=bid_std_dev,
                    reward_weights=reward_weights,
                    verify=verify,
                )

            return _gridnet(**make_kwargs)

        return _make

    if vec_env_class == "sync":
        envs = VecLuxEnv(num_envs, **make_kwargs)
    elif vec_env_class == "replay":
        envs = VecLuxReplayEnv(num_envs, **make_kwargs)
    else:
        envs = LuxAsyncVectorEnv([make(i) for i in range(n_envs)], copy=False)

    envs = HwcToChwObservation(envs)
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

    if additional_win_loss_reward:
        envs = AdditionalWinLossRewardWrapper(envs)
    if score_reward_kwargs:
        envs = ScoreRewardWrapper(envs, **score_reward_kwargs)

    return envs
