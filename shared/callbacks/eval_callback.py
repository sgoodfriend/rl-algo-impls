import os

from copy import deepcopy
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from torch.utils.tensorboard.writer import SummaryWriter
from typing import List, Optional, Union

from shared.callbacks.callback import Callback
from shared.policy.policy import Policy
from shared.stats import Episode, EpisodeAccumulator, EpisodesStats
from wrappers.vec_episode_recorder import VecEpisodeRecorder


class EvaluateAccumulator(EpisodeAccumulator):
    def __init__(self, num_envs: int, print_returns: bool = True):
        super().__init__(num_envs)
        self.print_returns = print_returns

    def on_done(self, ep_idx: int, episode: Episode) -> None:
        if self.print_returns:
            print(
                f"Episode {len(self.episodes)} | "
                f"Score {episode.score} | "
                f"Length {episode.length}"
            )


def evaluate(
    env: VecEnv,
    policy: Policy,
    n_episodes: int,
    render: bool = False,
    deterministic: bool = True,
    print_returns: bool = True,
) -> EpisodesStats:
    policy.eval()
    episodes = EvaluateAccumulator(env.num_envs, print_returns)

    obs = env.reset()
    while len(episodes) < n_episodes:
        act = policy.act(obs, deterministic=deterministic)
        obs, rew, done, _ = env.step(act)
        episodes.step(rew, done)
        if render:
            env.render()
    stats = EpisodesStats(episodes.episodes)
    if print_returns:
        print(stats)
    return stats


class EvalCallback(Callback):
    def __init__(
        self,
        policy: Policy,
        env: VecEnv,
        tb_writer: SummaryWriter,
        best_model_path: Optional[str] = None,
        step_freq: Union[int, float] = 50_000,
        n_episodes: int = 10,
        save_best: bool = True,
        deterministic: bool = True,
        record_best_videos: bool = True,
        video_env: Optional[VecEnv] = None,
        best_video_dir: Optional[str] = None,
        max_video_length: int = 3600,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.env = env
        self.tb_writer = tb_writer
        self.best_model_path = best_model_path
        self.step_freq = int(step_freq)
        self.n_episodes = n_episodes
        self.save_best = save_best
        self.deterministic = deterministic
        self.stats: List[EpisodesStats] = []
        self.best = None

        self.record_best_videos = record_best_videos
        assert video_env if record_best_videos else True
        self.video_env = video_env
        assert best_video_dir if record_best_videos else True
        self.best_video_dir = best_video_dir
        if best_video_dir:
            os.makedirs(best_video_dir, exist_ok=True)
        self.max_video_length = max_video_length

    def on_step(self, timesteps_elapsed: int = 1) -> bool:
        super().on_step(timesteps_elapsed)
        if self.timesteps_elapsed // self.step_freq >= len(self.stats):
            self.sync_vec_normalize(self.env)
            self.evaluate()
        return True

    def evaluate(
        self, n_episodes: Optional[int] = None, print_returns: Optional[bool] = None
    ) -> EpisodesStats:
        eval_stat = evaluate(
            self.env,
            self.policy,
            n_episodes or self.n_episodes,
            deterministic=self.deterministic,
            print_returns=print_returns or False,
        )
        self.policy.train(True)
        print(f"Eval Timesteps: {self.timesteps_elapsed} | {eval_stat}")

        self.stats.append(eval_stat)

        if not self.best or eval_stat >= self.best:
            strictly_better = not self.best or eval_stat > self.best
            self.best = eval_stat
            if self.save_best:
                assert self.best_model_path
                self.policy.save(self.best_model_path)
                print("Saved best model")
            if strictly_better and self.record_best_videos:
                assert self.video_env and self.best_video_dir
                self.sync_vec_normalize(self.video_env)
                video_wrapped = VecEpisodeRecorder(
                    self.video_env,
                    os.path.join(self.best_video_dir, str(self.timesteps_elapsed)),
                    max_video_length=self.max_video_length,
                )
                evaluate(
                    video_wrapped,
                    self.policy,
                    1,
                    deterministic=self.deterministic,
                    print_returns=False,
                )

        eval_stat.write_to_tensorboard(self.tb_writer, "eval", self.timesteps_elapsed)

        return eval_stat

    def sync_vec_normalize(self, destination_env: VecEnv) -> None:
        if self.policy.vec_normalize is not None:
            eval_env_wrapper = destination_env
            while isinstance(eval_env_wrapper, VecEnvWrapper):
                if isinstance(eval_env_wrapper, VecNormalize):
                    if hasattr(self.policy.vec_normalize, "obs_rms"):
                        eval_env_wrapper.obs_rms = deepcopy(
                            self.policy.vec_normalize.obs_rms
                        )
                    eval_env_wrapper.ret_rms = deepcopy(
                        self.policy.vec_normalize.ret_rms
                    )
                eval_env_wrapper = eval_env_wrapper.venv
