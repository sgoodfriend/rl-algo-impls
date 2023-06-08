from dataclasses import astuple
from typing import Optional

import gym
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.microrts.vec_env.microrts_socket_env import MicroRTSSocketEnv
from rl_algo_impls.microrts.wrappers.microrts_space_transform import (
    MicroRTSSpaceTransform,
)
from rl_algo_impls.microrts.wrappers.microrts_stats_recorder import (
    MicrortsStatsRecorder,
)
from rl_algo_impls.runner.config import Config, EnvHyperparams
from rl_algo_impls.wrappers.action_mask_wrapper import MicrortsMaskWrapper
from rl_algo_impls.wrappers.additional_win_loss_reward import (
    AdditionalWinLossRewardWrapper,
)
from rl_algo_impls.wrappers.episode_stats_writer import EpisodeStatsWriter
from rl_algo_impls.wrappers.hwc_to_chw_observation import HwcToChwObservation
from rl_algo_impls.wrappers.is_vector_env import IsVectorEnv
from rl_algo_impls.wrappers.score_reward_wrapper import ScoreRewardWrapper
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv


def make_microrts_env(
    config: Config,
    hparams: EnvHyperparams,
    training: bool = True,
    render: bool = False,
    normalize_load_path: Optional[str] = None,
    tb_writer: Optional[SummaryWriter] = None,
) -> VecEnv:
    from gym_microrts import microrts_ai

    from rl_algo_impls.microrts.vec_env.microrts_vec_env import (
        MicroRTSGridModeSharedMemVecEnv,
        MicroRTSGridModeVecEnv,
    )

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
        bots,
        self_play_kwargs,
        selfplay_bots,
        additional_win_loss_reward,
        map_paths,
        score_reward_kwargs,
        is_agent,
    ) = astuple(hparams)

    seed = config.seed(training=training)

    if not is_agent:
        make_kwargs = make_kwargs or {}
        self_play_kwargs = self_play_kwargs or {}
        if "num_selfplay_envs" not in make_kwargs:
            make_kwargs["num_selfplay_envs"] = 0
        if "num_bot_envs" not in make_kwargs:
            num_selfplay_envs = make_kwargs["num_selfplay_envs"]
            if num_selfplay_envs:
                num_bot_envs = (
                    n_envs
                    - make_kwargs["num_selfplay_envs"]
                    + self_play_kwargs.get("num_old_policies", 0)
                    + (len(selfplay_bots) if selfplay_bots else 0)
                )
            else:
                num_bot_envs = n_envs
            make_kwargs["num_bot_envs"] = num_bot_envs
        if "reward_weight" in make_kwargs:
            # Reward Weights:
            # WinLossRewardFunction
            # ResourceGatherRewardFunction
            # ProduceWorkerRewardFunction
            # ProduceBuildingRewardFunction
            # AttackRewardFunction
            # ProduceCombatUnitRewardFunction
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
            n_selfplay_historical_envs = self_play_kwargs.get("num_old_policies", 0)
            assert n_selfplay_historical_envs % (2 * len(map_paths)) == 0, (
                "Expect num_old_policies %d to be a multiple of twice len(map_paths) (2*%d)"
                % (
                    n_selfplay_historical_envs,
                    len(map_paths),
                )
            )
            for i in range(n_selfplay_historical_envs // 2):
                mp = map_paths[i % len(map_paths)]
                _map_paths.extend([mp, mp])

            n_selfplay_latest_envs = (
                make_kwargs["num_selfplay_envs"] - n_selfplay_historical_envs
            )
            assert (
                n_selfplay_latest_envs % len(map_paths) == 0
            ), "Expect num_selfplay_envs %d to be a multiple of len(map_paths) %d" % (
                n_selfplay_latest_envs,
                len(map_paths),
            )
            for i in range(n_selfplay_latest_envs):
                _map_paths.append(map_paths[i % len(map_paths)])

            n_bot_envs = make_kwargs["num_bot_envs"]
            assert (
                n_bot_envs % len(map_paths) == 0
            ), "Expect num_bot_envs %d to be a multiple of len(map_paths) %d" % (
                n_bot_envs,
                len(map_paths),
            )
            for i in range(n_bot_envs):
                _map_paths.append(map_paths[i % len(map_paths)])
            make_kwargs["map_paths"] = _map_paths
        make_kwargs["ai2s"] = ai2s

        envs = MicroRTSGridModeVecEnv(**make_kwargs)
    else:
        envs = MicroRTSSocketEnv()
    envs = MicroRTSSpaceTransform(
        envs, envs.utt, envs.height, envs.width, envs.partial_obs
    )
    envs = HwcToChwObservation(envs)
    envs = IsVectorEnv(envs)
    envs = MicrortsMaskWrapper(envs)

    if self_play_kwargs:
        if selfplay_bots:
            self_play_kwargs["selfplay_bots"] = selfplay_bots
        envs = SelfPlayWrapper(envs, config, **self_play_kwargs)

    if seed is not None:
        envs.action_space.seed(seed)
        envs.observation_space.seed(seed)

    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    if not is_agent:
        envs = MicrortsStatsRecorder(
            envs, config.algo_hyperparams.get("gamma", 0.99), bots
        )
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