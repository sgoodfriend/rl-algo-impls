import numpy as np
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)
from rl_algo_impls.wrappers.vector_wrapper import (
    VecEnvStepReturn,
    VectorWrapper,
    get_info,
)


class VecEpisodeRecorder(VectorWrapper):
    def __init__(
        self,
        env,
        tb_writer: AbstractSummaryWrapper,
        base_path: str,
        max_video_length: int = 3600,
        num_episodes: int = 1,
    ):
        super().__init__(env)
        self.tb_writer = tb_writer
        self.base_path = base_path
        self.max_video_length = max_video_length
        self.num_episodes = num_episodes
        self.video_recorder = None
        self.recorded_frames = 0
        self.num_completed = 0

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        obs, rew, terminations, truncations, infos = self.env.step(actions)
        dones = terminations | truncations
        # Using first env to record episodes
        if self.video_recorder:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if dones[0]:
                self.num_completed += 1
                ep_info = get_info(infos, "episode", 0)
                if ep_info:
                    episode_info = {
                        k: v.item() if hasattr(v, "item") else v
                        for k, v in ep_info.items()
                    }
                    if "episodes" not in self.video_recorder.metadata:
                        self.video_recorder.metadata["episodes"] = []
                    self.video_recorder.metadata["episodes"].append(episode_info)

            if (
                self.num_completed == self.num_episodes
                or self.recorded_frames > self.max_video_length
            ):
                self._close_video_recorder()
        return obs, rew, terminations, truncations, infos

    def reset(self, **kwargs):
        reset_return = self.env.reset(**kwargs)
        self._start_video_recorder()
        return reset_return

    def _start_video_recorder(self) -> None:
        self._close_video_recorder()

        self.video_recorder = VideoRecorder(
            self.env,
            base_path=self.base_path,
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1

    def _close_video_recorder(self) -> None:
        if self.video_recorder:
            self.video_recorder.close()
            self.tb_writer.log_video(
                self.video_recorder.path, self.video_recorder.frames_per_sec
            )
        self.video_recorder = None
        self.recorded_frames = 0
        self.num_completed = 0
