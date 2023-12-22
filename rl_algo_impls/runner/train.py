# Support for PyTorch mps mode (https://pytorch.org/docs/stable/notes/mps.html)
import logging
import os
import platform
import sys

from rl_algo_impls.checkpoints.checkpoints_manager import PolicyCheckpointsManager
from rl_algo_impls.lux.jux_verify import jux_verify_enabled
from rl_algo_impls.ppo.learning_rate_by_kl_divergence import LearningRateByKLDivergence
from rl_algo_impls.ppo.ppo import PPO
from rl_algo_impls.rollout.guided_learner_rollout import GuidedLearnerRolloutGenerator
from rl_algo_impls.rollout.random_guided_learner_rollout import (
    RandomGuidedLearnerRolloutGenerator,
)
from rl_algo_impls.rollout.reference_ai_rollout import ReferenceAIRolloutGenerator
from rl_algo_impls.rollout.sync_step_rollout import SyncStepRolloutGenerator
from rl_algo_impls.shared.agent_state import AgentState
from rl_algo_impls.shared.callbacks.self_play_callback import SelfPlayCallback
from rl_algo_impls.shared.callbacks.summary_wrapper import SummaryWrapper
from rl_algo_impls.shared.policy.policy import EnvSpaces

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import dataclasses
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import yaml
from torch.utils.tensorboard.writer import SummaryWriter

import wandb
from rl_algo_impls.runner.config import Config, EnvHyperparams, RunArgs
from rl_algo_impls.runner.running_utils import (
    ALGOS,
    DEFAULT_ROLLOUT_GENERATORS,
    get_device,
    hparam_dict,
    load_hyperparams,
    make_policy,
    plot_eval_callback,
    set_device_optimizations,
    set_seeds,
)
from rl_algo_impls.shared.callbacks.callback import Callback
from rl_algo_impls.shared.callbacks.eval_callback import EvalCallback
from rl_algo_impls.shared.callbacks.hyperparam_transitions import HyperparamTransitions
from rl_algo_impls.shared.callbacks.reward_decay_callback import RewardDecayCallback
from rl_algo_impls.shared.stats import EpisodesStats
from rl_algo_impls.shared.vec_env import make_env, make_eval_env
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper
from rl_algo_impls.wrappers.vector_wrapper import find_wrapper


@dataclass
class TrainArgs(RunArgs):
    wandb_project_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: Sequence[str] = dataclasses.field(default_factory=list)
    wandb_group: Optional[str] = None


def train(args: TrainArgs):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info(args)
    hyperparams = load_hyperparams(args.algo, args.env)
    logging.info(hyperparams)
    config = Config(args, hyperparams, os.getcwd())

    wandb_enabled = bool(args.wandb_project_name)
    if wandb_enabled:
        wandb.tensorboard.patch(
            root_logdir=config.tensorboard_summary_path, pytorch=True
        )
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=asdict(hyperparams),
            name=config.run_name(),
            monitor_gym=True,
            save_code=True,
            tags=args.wandb_tags,
            group=args.wandb_group,
        )
        wandb.config.update(args)

    tb_writer = SummaryWrapper(SummaryWriter(config.tensorboard_summary_path))

    set_seeds(args.seed)

    checkpoints_manager = (
        PolicyCheckpointsManager(**config.hyperparams.checkpoints_kwargs)
        if config.hyperparams.checkpoints_kwargs
        else None
    )

    agent_state = AgentState()
    env = make_env(
        config,
        EnvHyperparams(**config.env_hyperparams),
        agent_state,
        tb_writer=tb_writer,
        checkpoints_manager=checkpoints_manager,
    )
    device = get_device(config, env)
    set_device_optimizations(device, **config.device_hyperparams)
    policy = make_policy(
        config,
        EnvSpaces.from_vec_env(env),
        device,
        agent_state,
        **config.policy_hyperparams,
    )
    algo = ALGOS[args.algo](
        policy, device, tb_writer, **config.algo_hyperparams(checkpoints_manager)
    )
    agent_state.algo = algo

    num_parameters = policy.num_parameters()
    num_trainable_parameters = policy.num_trainable_parameters()
    if wandb_enabled:
        wandb.run.summary["num_parameters"] = num_parameters  # type: ignore
        wandb.run.summary["num_trainable_parameters"] = num_trainable_parameters  # type: ignore
    else:
        print(
            f"num_parameters = {num_parameters} ; "
            f"num_trainable_parameters = {num_trainable_parameters}"
        )

    self_play_wrapper = find_wrapper(env, SelfPlayWrapper)
    eval_env = make_eval_env(
        config,
        EnvHyperparams(**config.env_hyperparams),
        agent_state,
        self_play_wrapper=self_play_wrapper,
        checkpoints_manager=checkpoints_manager,
    )
    video_env = make_eval_env(
        config,
        EnvHyperparams(**config.env_hyperparams),
        agent_state,
        override_hparams={"n_envs": 1},
        self_play_wrapper=self_play_wrapper,
        checkpoints_manager=checkpoints_manager,
    )
    eval_callback = EvalCallback(
        agent_state,
        eval_env,
        tb_writer,
        best_model_path=config.model_dir_path(best=True),
        **config.eval_callback_params(),
        video_env=video_env,
        video_dir=config.videos_path,
        additional_keys_to_log=config.additional_keys_to_log,
        wandb_enabled=wandb_enabled,
        latest_model_path=config.model_dir_path(best=False),
        checkpoints_manager=checkpoints_manager,
    )
    callbacks: List[Callback] = [eval_callback]
    if config.hyperparams.reward_decay_callback:
        callbacks.append(
            RewardDecayCallback(
                config, env, **(config.hyperparams.reward_decay_callback_kwargs or {})
            )
        )
    if self_play_wrapper:
        callbacks.append(SelfPlayCallback(policy, self_play_wrapper))

    rollout_hyperparams = {**config.rollout_hyperparams}
    subaction_mask = config.policy_hyperparams.get("subaction_mask", None)
    if subaction_mask is not None:
        rollout_hyperparams["subaction_mask"] = subaction_mask
    if config.rollout_type:
        if config.rollout_type == "sync":
            rollout_generator_cls = SyncStepRolloutGenerator
        elif config.rollout_type == "reference":
            rollout_generator_cls = ReferenceAIRolloutGenerator
        elif config.rollout_type in {"guided", "guided_random"}:
            if config.rollout_type == "guided_random":
                rollout_generator_cls = RandomGuidedLearnerRolloutGenerator
            elif config.rollout_type == "guided":
                rollout_generator_cls = GuidedLearnerRolloutGenerator
            else:
                raise ValueError(f"{config.rollout_type} not recognized rollout_type")
            guide_policy_hyperparams = {
                **config.policy_hyperparams,
                **rollout_hyperparams.get("guide_policy", {}),
            }
            rollout_hyperparams["guide_policy"] = make_policy(
                config, EnvSpaces.from_vec_env(env), device, **guide_policy_hyperparams
            )
        else:
            raise ValueError(f"{config.rollout_type} not recognized rollout_type")
    else:
        rollout_generator_cls = DEFAULT_ROLLOUT_GENERATORS[args.algo]

    rollout_generator = rollout_generator_cls(
        policy,  # type: ignore
        env,
        **rollout_hyperparams,
    )
    rollout_generator.prepare()
    lr_by_kl_callback = None
    if config.hyperparams.lr_by_kl_kwargs:
        assert isinstance(
            algo, PPO
        ), f"lr_by_kl_kwargs only supported for PPO, not {algo.__class__.__name__}"
        lr_by_kl_callback = LearningRateByKLDivergence(
            algo,
            **config.hyperparams.lr_by_kl_kwargs,
        )
        callbacks.append(lr_by_kl_callback)
    if config.hyperparams.hyperparam_transitions_kwargs:
        callbacks.append(
            HyperparamTransitions(
                config,
                env,
                algo,
                rollout_generator,
                **config.hyperparams.hyperparam_transitions_kwargs,
                lr_by_kl_callback=lr_by_kl_callback,
            )
        )

    if jux_verify_enabled(eval_env):
        eval_callback.generate_video()
    algo.learn(config.n_timesteps, rollout_generator, callbacks=callbacks)

    policy.save(config.model_dir_path(best=False))

    eval_stats = eval_callback.evaluate(n_episodes=10, print_returns=True)

    plot_eval_callback(eval_callback, tb_writer, config.run_name())

    log_dict: Dict[str, Any] = {
        "eval": eval_stats._asdict(),
    }
    hparam_metric_dict = {
        "hparam/last_mean": eval_stats.score.mean,
        "hparam/last_result": eval_stats.score.mean - eval_stats.score.std,
    }
    if eval_callback.best:
        best_eval_stats: EpisodesStats = eval_callback.best
        log_dict["best_eval"] = best_eval_stats._asdict()
        hparam_metric_dict.update(
            {
                "hparam/best_mean": best_eval_stats.score.mean,
                "hparam/best_result": best_eval_stats.score.mean
                - best_eval_stats.score.std,
            }
        )
    log_dict.update(asdict(hyperparams))
    log_dict.update(vars(args))
    with open(config.logs_path, "a") as f:
        yaml.dump({config.run_name(): log_dict}, f)

    tb_writer.add_hparams(
        hparam_dict(hyperparams, vars(args)),
        hparam_metric_dict,
        None,
        config.run_name(),
    )

    tb_writer.close()

    if wandb_enabled:
        shutil.make_archive(
            os.path.join(wandb.run.dir, config.model_dir_name()),  # type: ignore
            "zip",
            config.model_dir_path(),
        )
        wandb.finish()
