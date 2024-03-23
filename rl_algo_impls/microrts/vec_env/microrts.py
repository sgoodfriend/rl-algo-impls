from dataclasses import astuple
from typing import Optional

import numpy as np
from gymnasium.experimental.wrappers.vector.record_episode_statistics import (
    RecordEpisodeStatisticsV0,
)

from rl_algo_impls.microrts.vec_env.microrts_socket_env import MicroRTSSocketEnv
from rl_algo_impls.microrts.vec_env.microrts_space_transform import (
    MicroRTSSpaceTransform,
)
from rl_algo_impls.microrts.wrappers.microrts_stats_recorder import (
    MicrortsStatsRecorder,
)
from rl_algo_impls.runner.config import Config
from rl_algo_impls.runner.env_hyperparams import EnvHyperparams
from rl_algo_impls.shared.data_store.data_store_view import VectorEnvDataStoreView
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.wrappers.action_mask_stats_recorder import ActionMaskStatsRecorder
from rl_algo_impls.wrappers.action_mask_wrapper import MicrortsMaskWrapper
from rl_algo_impls.wrappers.additional_win_loss_reward import (
    AdditionalWinLossRewardWrapper,
)
from rl_algo_impls.wrappers.episode_stats_writer import EpisodeStatsWriter
from rl_algo_impls.wrappers.hwc_to_chw_observation import HwcToChwVectorObservation
from rl_algo_impls.wrappers.info_rewards_wrapper import InfoRewardsWrapper
from rl_algo_impls.wrappers.is_vector_env import IsVectorEnv
from rl_algo_impls.wrappers.normalize import NormalizeObservation, NormalizeReward
from rl_algo_impls.wrappers.play_checkpoints_wrapper import PlayCheckpointsWrapper
from rl_algo_impls.wrappers.score_reward_wrapper import ScoreRewardWrapper
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper
from rl_algo_impls.wrappers.vector_wrapper import VectorEnv


def make_microrts_env(
    config: Config,
    hparams: EnvHyperparams,
    data_store_view: VectorEnvDataStoreView,
    training: bool = True,
    render: bool = False,
    tb_writer: Optional[AbstractSummaryWrapper] = None,
    **kwargs,
) -> VectorEnv:
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
        _,  # video_step_interval
        _,  # initial_steps_to_truncate
        _,  # clip_atari_rewards
        normalize_type,
        _,  # mask_actions
        bots,
        self_play_kwargs,
        selfplay_bots,
        additional_win_loss_reward,
        map_paths,
        score_reward_kwargs,
        is_agent,
        valid_sizes,
        paper_planes_sizes,
        fixed_size,
        terrain_overrides,
        time_budget_ms,
        video_frames_per_second,
        _,  # reference_bot,
        play_checkpoints_kwargs,
        _,  # additional_win_loss_smoothing_factor,
        info_rewards,
        disallow_no_op,
    ) = astuple(hparams)

    seed = config.seed(training=training)

    if not is_agent:
        from rl_algo_impls.microrts import microrts_ai
        from rl_algo_impls.microrts.vec_env.microrts_vec_env import (
            MicroRTSGridModeVecEnv,
        )

        make_kwargs = make_kwargs or {}

        self_play_kwargs = self_play_kwargs or {}
        play_checkpoints_kwargs = play_checkpoints_kwargs or {}
        assert not (
            bool(self_play_kwargs) and bool(play_checkpoints_kwargs)
        ), "Cannot have both self_play_kwargs and play_checkpoints_kwargs"
        num_checkpoint_envs = self_play_kwargs.get(
            "num_old_policies", 0
        ) or play_checkpoints_kwargs.get("n_envs_against_checkpoints", 0)

        if "num_selfplay_envs" not in make_kwargs:
            make_kwargs["num_selfplay_envs"] = 0
        if "num_bot_envs" not in make_kwargs:
            num_selfplay_envs = make_kwargs["num_selfplay_envs"]
            if num_selfplay_envs:
                num_bot_envs = (
                    n_envs
                    - make_kwargs["num_selfplay_envs"]
                    + num_checkpoint_envs
                    + (len(selfplay_bots) if selfplay_bots else 0)
                )
            else:
                num_bot_envs = n_envs
            make_kwargs["num_bot_envs"] = num_bot_envs
        if "reward_weight" in make_kwargs:
            # Reward Weights:
            # RAIWinLossRewardFunction
            # ResourceGatherRewardFunction
            # ProduceWorkerRewardFunction
            # ProduceBuildingRewardFunction
            # AttackRewardFunction
            # ProduceLightUnitRewardFunction
            # ProduceHeavyUnitRewardFunction
            # ProduceRangedUnitRewardFunction
            # ScoreRewardFunction
            make_kwargs["reward_weight"] = np.array(make_kwargs["reward_weight"])
        if bots:
            ai2s = []
            for ai_name, n in bots.items():
                for _ in range(n):
                    if len(ai2s) >= make_kwargs["num_bot_envs"]:
                        break
                    ai = getattr(microrts_ai, ai_name)
                    assert ai, f"{ai_name} not in microrts_ai"
                    ai2s.append(ai)
        else:
            ai2s = [microrts_ai.randomAI for _ in range(make_kwargs["num_bot_envs"])]
        if map_paths:
            _map_paths = []
            n_selfplay_historical_envs = num_checkpoint_envs * 2
            assert n_selfplay_historical_envs % (4 * len(map_paths)) == 0, (
                "Expect num_checkpoint_envs %d to be a multiple of 2 len(map_paths) (2*%d)"
                % (
                    num_checkpoint_envs,
                    len(map_paths),
                )
            )
            for i in range(n_selfplay_historical_envs // 4):
                mp = map_paths[i % len(map_paths)]
                _map_paths.extend([mp] * 4)

            n_selfplay_latest_envs = (
                make_kwargs["num_selfplay_envs"] - n_selfplay_historical_envs
            )
            assert n_selfplay_latest_envs % (2 * len(map_paths)) == 0, (
                "Expect num_selfplay_envs %d to be a multiple of twice len(map_paths) (2*%d)"
                % (
                    n_selfplay_latest_envs,
                    len(map_paths),
                )
            )
            for i in range(n_selfplay_latest_envs // 2):
                mp = map_paths[i % len(map_paths)]
                _map_paths.extend([mp] * 2)

            if bots:
                for ai_name, n in bots.items():
                    assert (
                        n % len(map_paths) == 0
                    ), f"Expect number of {ai_name} bots ({n}) to be a multiple of len(map_paths) ({len(map_paths)})"
                    for mp in map_paths:
                        _map_paths.extend([mp] * (n // len(map_paths)))
            else:
                n_bot_envs = make_kwargs["num_bot_envs"]
                assert (
                    n_bot_envs % len(map_paths) == 0
                ), "Expect num_bot_envs %d to be a multiple of len(map_paths) (%d)" % (
                    n_bot_envs,
                    len(map_paths),
                )
                for mp in map_paths:
                    _map_paths.extend([mp] * (n_bot_envs // len(map_paths)))

            make_kwargs["map_paths"] = _map_paths
        make_kwargs["ai2s"] = ai2s

        envs = MicroRTSGridModeVecEnv(
            **make_kwargs, video_frames_per_second=video_frames_per_second
        )
    else:
        envs = MicroRTSSocketEnv.singleton(time_budget_ms=time_budget_ms)
    envs = MicroRTSSpaceTransform(
        envs,
        valid_sizes=valid_sizes,
        paper_planes_sizes=paper_planes_sizes,
        fixed_size=fixed_size,
        terrain_overrides=terrain_overrides,
        disallow_no_op=disallow_no_op,
    )
    envs = HwcToChwVectorObservation(envs)
    envs = IsVectorEnv(envs)
    envs = MicrortsMaskWrapper(envs)

    if self_play_kwargs:
        if selfplay_bots:
            self_play_kwargs["selfplay_bots"] = selfplay_bots
        envs = SelfPlayWrapper(envs, config, **self_play_kwargs)
    if play_checkpoints_kwargs:
        envs = PlayCheckpointsWrapper(envs, **play_checkpoints_kwargs)
        data_store_view.add_checkpoint_policy_delegate(envs)
    if seed is not None:
        envs.action_space.seed(seed)
        envs.observation_space.seed(seed)

    envs = RecordEpisodeStatisticsV0(envs)
    if not is_agent:
        envs = MicrortsStatsRecorder(
            envs,
            bots,
        )
    envs = ActionMaskStatsRecorder(envs)
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
    if info_rewards:
        envs = InfoRewardsWrapper(envs, **info_rewards, flatten_info=True)
    if normalize:
        if normalize_type is None:
            normalize_type = "gymlike"
        if normalize_type == "gymlike":
            if normalize_kwargs.get("norm_obs", True):
                envs = NormalizeObservation(
                    envs,
                    data_store_view,
                    training=training,
                    clip=normalize_kwargs.get("clip_obs", 10.0),
                    normalize_axes=tuple(
                        normalize_kwargs.get("normalize_axes", tuple())
                    ),
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
                    data_store_view,
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
