import itertools
import logging
import os
import shutil
from time import perf_counter
from typing import Deque, Dict, List, Optional, Union

import numpy as np

from rl_algo_impls.checkpoints.checkpoints_manager import PolicyCheckpointsManager
from rl_algo_impls.lux.jux_verify import jux_verify_enabled
from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.callbacks import Callback
from rl_algo_impls.shared.callbacks.summary_wrapper import SummaryWrapper
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.stats import Episode, EpisodeAccumulator, EpisodesStats
from rl_algo_impls.shared.tensor_utils import batch_dict_keys
from rl_algo_impls.wrappers.vec_episode_recorder import VecEpisodeRecorder
from rl_algo_impls.wrappers.vector_wrapper import VectorEnv


class EvaluateAccumulator(EpisodeAccumulator):
    def __init__(
        self,
        num_envs: int,
        goal_episodes: int,
        print_returns: bool = True,
        ignore_first_episode: bool = False,
        additional_keys_to_log: Optional[List[str]] = None,
    ):
        super().__init__(num_envs)
        self.completed_episodes_by_env_idx = [[] for _ in range(num_envs)]
        self.goal_episodes_per_env = int(np.ceil(goal_episodes / num_envs))
        self.print_returns = print_returns
        if ignore_first_episode:
            first_done = set()

            def should_record_done(idx: int) -> bool:
                has_done_first_episode = idx in first_done
                first_done.add(idx)
                return has_done_first_episode

            self.should_record_done = should_record_done
        else:
            self.should_record_done = lambda idx: True
        self.additional_keys_to_log = additional_keys_to_log

    def on_done(self, ep_idx: int, episode: Episode, info: Dict) -> None:
        if self.additional_keys_to_log:
            episode.info = {k: info[k] for k in self.additional_keys_to_log}
        if (
            self.should_record_done(ep_idx)
            and len(self.completed_episodes_by_env_idx[ep_idx])
            >= self.goal_episodes_per_env
        ):
            return
        self.completed_episodes_by_env_idx[ep_idx].append(episode)
        if self.print_returns:
            print(
                f"Episode {len(self)} | "
                f"Score {episode.score} | "
                f"Length {episode.length}"
            )

    def __len__(self) -> int:
        return sum(len(ce) for ce in self.completed_episodes_by_env_idx)

    @property
    def episodes(self) -> List[Episode]:
        return list(itertools.chain(*self.completed_episodes_by_env_idx))

    def is_done(self) -> bool:
        return all(
            len(ce) == self.goal_episodes_per_env
            for ce in self.completed_episodes_by_env_idx
        )


def evaluate(
    env: VectorEnv,
    policy: Policy,
    n_episodes: int,
    render: bool = False,
    deterministic: bool = True,
    print_returns: bool = True,
    ignore_first_episode: bool = False,
    additional_keys_to_log: Optional[List[str]] = None,
    score_function: str = "mean-std",
) -> EpisodesStats:
    policy.sync_normalization(env)
    policy.eval()

    episodes = EvaluateAccumulator(
        env.num_envs,
        n_episodes,
        print_returns,
        ignore_first_episode,
        additional_keys_to_log=additional_keys_to_log,
    )

    obs, _ = env.reset()
    get_action_mask = getattr(env, "get_action_mask", None)

    old_vec_jux_state = None
    old_jux_obs = None
    old_jux_action_mask = None
    jux_env_batch = None
    if jux_verify_enabled(env):
        import jax
        import jax.numpy as jnp
        from jux.env import JuxEnvBatch
        from jux.state.state import State as JuxState

        from rl_algo_impls.lux.jux.actions import step_unified
        from rl_algo_impls.lux.jux.observation import observation_and_action_mask
        from rl_algo_impls.lux.vec_env.jux_vector_env import JuxVectorEnv

        lux_gridnet = env.unwrapped.envs[0]
        action_mask = get_action_mask()

        jux_env = policy.env.unwrapped
        assert isinstance(jux_env, JuxVectorEnv)
        lux_state = lux_gridnet.state
        jux_state = JuxState.from_lux(lux_state, jux_env.buf_cfg)
        old_vec_jux_state = jax.tree_util.tree_map(
            lambda a: jnp.expand_dims(a, 0), jux_state
        )
        old_jux_obs, old_jux_action_mask = observation_and_action_mask(
            old_vec_jux_state, jux_env.env_cfg, jux_env.buf_cfg, jux_env.agent_cfg
        )
        assert np.allclose(obs[0], old_jux_obs[1])
        assert np.allclose(
            action_mask[0]["per_position"], old_jux_action_mask["per_position"][1]
        )
        assert np.allclose(
            action_mask[0]["pick_position"], old_jux_action_mask["pick_position"][1]
        )

        jux_env_batch = JuxEnvBatch(jux_env.env_cfg, jux_env.buf_cfg)

    while not episodes.is_done():
        act = policy.act(
            obs,
            deterministic=deterministic,
            action_masks=batch_dict_keys(get_action_mask())
            if get_action_mask
            else None,
        )
        obs, rew, terminations, truncations, info = env.step(act)

        if jux_verify_enabled(env):
            import jax
            import jax.numpy as jnp
            from jux.state.state import State as JuxState

            from rl_algo_impls.lux.jux.actions import step_unified
            from rl_algo_impls.lux.jux.observation import observation_and_action_mask
            from rl_algo_impls.lux.vec_env.jux_vector_env import JuxVectorEnv

            assert old_vec_jux_state is not None
            assert old_jux_obs is not None
            assert old_jux_action_mask is not None
            assert jux_env_batch is not None
            lux_gridnet = env.unwrapped.envs[0]
            action_mask = get_action_mask()

            jux_env = policy.env.unwrapped
            assert isinstance(jux_env, JuxVectorEnv)
            lux_state = lux_gridnet.env.state
            jux_state = JuxState.from_lux(lux_state, jux_env.buf_cfg)
            vec_jux_state = jax.tree_util.tree_map(
                lambda a: jnp.expand_dims(a, 0), jux_state
            )
            jux_obs, jux_action_mask = observation_and_action_mask(
                vec_jux_state, jux_env.env_cfg, jux_env.buf_cfg, jux_env.agent_cfg
            )

            assert np.allclose(obs[0], jux_obs[1])
            assert np.allclose(
                action_mask[0]["per_position"], jux_action_mask["per_position"][1]
            )
            assert np.allclose(
                action_mask[0]["pick_position"], jux_action_mask["pick_position"][1]
            )

            gridnet_act = lux_gridnet._prior_action
            jax_actions = {
                "pick_position": jnp.array(
                    tuple(a["pick_position"] for a in gridnet_act), dtype=jnp.int16
                ).reshape(1, 2, -1),
                "per_position": jnp.array(
                    tuple(a["per_position"] for a in gridnet_act), dtype=jnp.int8
                ).reshape(1, 2, jux_env.map_size, jux_env.map_size, -1),
            }
            (
                jux_run_state,
                jux_run_obs,
                jux_run_action_mask,
                _,
                jux_run_done,
                _,
            ) = step_unified(
                jux_env_batch,
                old_vec_jux_state,
                jux_env.env_cfg,
                jux_env.buf_cfg,
                jax_actions,
                old_jux_obs,
                jux_env.agent_cfg,
            )

            assert jux_run_done[1] == terminations[0]
            if not terminations[0]:
                assert np.allclose(obs[0], jux_run_obs[1])
                assert np.allclose(
                    action_mask[0]["per_position"],
                    jux_run_action_mask["per_position"][1],
                )
                assert np.allclose(
                    action_mask[0]["pick_position"],
                    jux_run_action_mask["pick_position"][1],
                )

                old_vec_jux_state = jux_run_state
            else:
                old_vec_jux_state = vec_jux_state
            old_jux_obs = jux_obs
            old_jux_action_mask = jux_action_mask

        done = terminations | truncations
        episodes.step(rew, done, info)
        if render:
            env.render()
    stats = EpisodesStats(
        episodes.episodes,
        score_function=score_function,
    )
    if print_returns:
        print(stats)
    return stats


class EvalCallback(Callback):
    prior_policies: Optional[Deque[Policy]]

    def __init__(
        self,
        policy: Policy,
        algo: Algorithm,
        env: VectorEnv,
        tb_writer: SummaryWrapper,
        best_model_path: Optional[str] = None,
        step_freq: Union[int, float] = 50_000,
        n_episodes: int = 10,
        save_best: bool = True,
        deterministic: bool = True,
        only_record_video_on_best: bool = True,
        video_env: Optional[VectorEnv] = None,
        video_dir: Optional[str] = None,
        max_video_length: int = 9000,
        ignore_first_episode: bool = False,
        additional_keys_to_log: Optional[List[str]] = None,
        score_function: str = "mean-std",
        wandb_enabled: bool = False,
        score_threshold: Optional[float] = None,
        skip_evaluate_at_start: bool = False,
        only_checkpoint_initial_policy: bool = False,
        only_checkpoint_best_policies: bool = False,
        latest_model_path: Optional[str] = None,
        checkpoints_manager: Optional[PolicyCheckpointsManager] = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.algo = algo
        self.env = env
        self.tb_writer = tb_writer
        self.best_model_path = best_model_path
        self.step_freq = int(step_freq)
        self.n_episodes = n_episodes
        self.save_best = save_best
        self.deterministic = deterministic
        self.stats: List[EpisodesStats] = []
        self.best = None

        self.only_record_video_on_best = only_record_video_on_best
        self.max_video_length = max_video_length
        assert (video_env is not None) == (
            video_dir is not None
        ), f"video_env ({video_env}) and video_dir ({video_dir}) must be set or unset together"
        self.video_dir = video_dir
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
            self.video_env = VecEpisodeRecorder(
                video_env,
                video_dir,  # This is updated when a video is actually created
                max_video_length=self.max_video_length,
            )
        else:
            self.video_env = None

        self.ignore_first_episode = ignore_first_episode
        self.additional_keys_to_log = additional_keys_to_log
        self.score_function = score_function
        self.wandb_enabled = wandb_enabled
        self.score_threshold = score_threshold
        self.skip_evaluate_at_start = skip_evaluate_at_start
        self.latest_model_path = latest_model_path

        self.checkpoints_manager = checkpoints_manager
        if checkpoints_manager:
            if only_checkpoint_initial_policy:
                assert (
                    checkpoints_manager.history_size == 1
                ), f"checkpoints_manager's history_size must be 1 if only_checkpoint_initial_policy is True"
            self.only_checkpoint_initial_policy = only_checkpoint_initial_policy
            self.only_checkpoint_best_policies = only_checkpoint_best_policies
            self.checkpoint_policy(True)
        else:
            self.prior_policies = None

    def on_step(self, timesteps_elapsed: int = 1, **kwargs) -> bool:
        super().on_step(timesteps_elapsed)
        desired_num_stats = self.timesteps_elapsed // self.step_freq
        if not self.skip_evaluate_at_start:
            desired_num_stats += 1
        if desired_num_stats > len(self.stats):
            self.evaluate()
        return True

    def evaluate(
        self, n_episodes: Optional[int] = None, print_returns: Optional[bool] = None
    ) -> EpisodesStats:
        start_time = perf_counter()
        eval_stat = evaluate(
            self.env,
            self.policy,
            n_episodes or self.n_episodes,
            deterministic=self.deterministic,
            print_returns=print_returns or False,
            ignore_first_episode=self.ignore_first_episode,
            additional_keys_to_log=self.additional_keys_to_log,
            score_function=self.score_function,
        )
        end_time = perf_counter()
        self.tb_writer.add_scalar(
            "eval/steps_per_second",
            eval_stat.length.sum() / (end_time - start_time),
        )
        self.policy.train(True)
        print(f"Eval Timesteps: {self.timesteps_elapsed} | {eval_stat}")

        self.stats.append(eval_stat)

        if self.score_threshold is not None:
            is_best = eval_stat.score.score() >= self.score_threshold
            strictly_better = eval_stat.score.score() > self.score_threshold
        else:
            is_best = not self.best or eval_stat >= self.best
            strictly_better = not self.best or eval_stat > self.best

        if self.latest_model_path:
            self.policy.save(self.latest_model_path)
            self.algo.save(self.latest_model_path)
        if is_best:
            self.best = eval_stat
            if self.save_best:
                assert self.best_model_path
                self.policy.save(self.best_model_path)
                self.algo.save(self.best_model_path)
                print("Saved best model")
                if self.wandb_enabled:
                    import wandb

                    best_model_name = os.path.split(self.best_model_path)[-1]
                    shutil.make_archive(
                        os.path.join(wandb.run.dir, best_model_name),  # type: ignore
                        "zip",
                        self.best_model_path,
                    )
            self.best.write_to_tensorboard(self.tb_writer, "best_eval")
        if self.video_env and (not self.only_record_video_on_best or strictly_better):
            self.generate_video()

        eval_stat.write_to_tensorboard(self.tb_writer, "eval")
        self.checkpoint_policy(is_best)
        return eval_stat

    def checkpoint_policy(self, is_best: bool):
        if self.checkpoints_manager is None:
            return
        if len(self.checkpoints_manager.checkpoints) > 0:
            if self.only_checkpoint_initial_policy:
                return
            if self.only_checkpoint_best_policies and not is_best:
                return
            else:
                logging.info(f"Checkpointing best policy at {self.timesteps_elapsed}")
        self.checkpoints_manager.create_checkpoint(self.policy)

    def generate_video(self) -> None:
        assert self.video_env and self.video_dir
        best_video_base_path = os.path.join(self.video_dir, str(self.timesteps_elapsed))
        self.video_env.base_path = best_video_base_path
        video_stats = evaluate(
            self.video_env,
            self.policy,
            1,
            deterministic=self.deterministic,
            print_returns=False,
            score_function=self.score_function,
        )
        print(f"Saved video: {video_stats}")
