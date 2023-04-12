from typing import Tuple, Union

import gym
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from rl_algo_impls.wrappers.vectorable_wrapper import VectorableWrapper

ObsType = Union[np.ndarray, dict]
ActType = Union[int, float, np.ndarray, dict]


class EpisodeRecordVideo(VectorableWrapper):
    def __init__(
        self,
        env: gym.Env,
        video_path_prefix: str,
        step_increment: int = 1,
        video_step_interval: int = 1_000_000,
        max_video_length: int = 3600,
    ) -> None:
        super().__init__(env)
        self.video_path_prefix = video_path_prefix
        self.step_increment = step_increment
        self.video_step_interval = video_step_interval
        self.max_video_length = max_video_length
        self.total_steps = 0
        self.next_record_video_step = 0
        self.video_recorder = None
        self.recorded_frames = 0

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, rew, done, info = self.env.step(action)
        self.total_steps += self.step_increment
        # Using first env to record episodes
        if self.video_recorder:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if info.get("episode"):
                episode_info = {
                    k: v.item() if hasattr(v, "item") else v
                    for k, v in info["episode"].items()
                }
                self.video_recorder.metadata["episode"] = episode_info
            if self.recorded_frames > self.max_video_length:
                self._close_video_recorder()
        return obs, rew, done, info

    def reset(self, **kwargs) -> ObsType:
        obs = self.env.reset(**kwargs)
        if self.video_recorder:
            self._close_video_recorder()
        elif self.total_steps >= self.next_record_video_step:
            self._start_video_recorder()
        return obs

    def _start_video_recorder(self) -> None:
        self._close_video_recorder()

        video_path = f"{self.video_path_prefix}-{self.next_record_video_step}"
        self.video_recorder = VideoRecorder(
            self.env,
            base_path=video_path,
            metadata={"step": self.total_steps},
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.next_record_video_step += self.video_step_interval

    def _close_video_recorder(self) -> None:
        if self.video_recorder:
            self.video_recorder.close()
        self.video_recorder = None
        self.recorded_frames = 0
