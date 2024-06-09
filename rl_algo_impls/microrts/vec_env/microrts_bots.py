from dataclasses import astuple
from typing import Optional

import numpy as np
from gymnasium.experimental.wrappers.vector.record_episode_statistics import (
    RecordEpisodeStatisticsV0,
)

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
from rl_algo_impls.wrappers.is_vector_env import IsVectorEnv
from rl_algo_impls.wrappers.score_reward_wrapper import ScoreRewardWrapper
from rl_algo_impls.wrappers.vector_wrapper import VectorEnv


def make_microrts_bots_env(
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
        _,  # normalize
        _,  # normalize_kwargs,
        rolling_length,
        _,  # video_step_interval
        _,  # initial_steps_to_truncate
        _,  # clip_atari_rewards
        _,  # normalize_type
        _,  # mask_actions
        bots,
        _,  # self_play_kwargs,
        _,  # selfplay_bots,
        additional_win_loss_reward,
        map_paths,
        score_reward_kwargs,
        _,  # is_agent,
        valid_sizes,
        paper_planes_sizes,
        fixed_size,
        terrain_overrides,
        _,  # time_budget_ms,
        video_frames_per_second,
        reference_bot,
        _,  # play_checkpoints_kwargs,
        _,  # additional_win_loss_smoothing_factor,
        _,  # info_rewards,
        _,  # disallow_no_op,
        ignore_mask,
    ) = astuple(hparams)

    seed = config.seed(training=training)

    from rl_algo_impls.microrts import microrts_ai
    from rl_algo_impls.microrts.vec_env.microrts_bot_vec_env import (
        MicroRTSBotGridVecEnv,
    )

    make_kwargs = make_kwargs or {}
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

    assert bots, f"Must specify opponent bots"

    assert reference_bot, f"Must specify reference_bot"
    ref_ai = getattr(microrts_ai, reference_bot)
    assert ref_ai, f"{reference_bot} not in microrts_ai"
    if map_paths:
        _map_paths = []
        _ais = []
        for ai_name, n in bots.items():
            modulus = len(map_paths) * (1 if ai_name == reference_bot else 2)
            assert (
                n % modulus == 0
            ), f"Expect number of {ai_name} bots ({n}) to be a multiple of {modulus}"
            env_per_map = 2 * n // len(map_paths)
            opp_ai = getattr(microrts_ai, ai_name)
            for mp in map_paths:
                _map_paths.extend([mp] * env_per_map)
                for i in range(env_per_map // 2):
                    _ais.extend([opp_ai, ref_ai] if i % 2 else [ref_ai, opp_ai])
        make_kwargs["map_paths"] = _map_paths
        make_kwargs["ais"] = _ais
    else:
        _ais = []
        for ai_name, n in bots.items():
            for i in range(n):
                opp_ai = getattr(microrts_ai, ai_name)
                assert opp_ai, f"{ai_name} not in microrts_ai"
                _ais.extend([opp_ai, ref_ai] if i % 2 else [ref_ai, opp_ai])
        make_kwargs["ais"] = _ais
    make_kwargs["reference_indexes"] = [
        idx for idx, ai in enumerate(make_kwargs["ais"]) if ai == ref_ai
    ]

    envs = MicroRTSBotGridVecEnv(
        **make_kwargs, video_frames_per_second=video_frames_per_second
    )

    envs = MicroRTSSpaceTransform(
        envs,
        valid_sizes=valid_sizes,
        paper_planes_sizes=paper_planes_sizes,
        fixed_size=fixed_size,
        terrain_overrides=terrain_overrides,
        ignore_mask=ignore_mask,
    )
    envs = HwcToChwVectorObservation(envs)
    envs = IsVectorEnv(envs)
    envs = MicrortsMaskWrapper(envs)

    if seed is not None:
        envs.action_space.seed(seed)
        envs.observation_space.seed(seed)

    envs = RecordEpisodeStatisticsV0(envs)
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

    return envs
