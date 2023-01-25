import numpy as np

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvStepReturn,
    VecEnvWrapper,
    VecEnvObs,
)
from typing import Optional


class VecSingleImageWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv):
        super().__init__(venv)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        images = self.venv.get_images()
        return images[0]

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()

    def reset(self) -> VecEnvObs:
        return self.venv.reset()


class VecSingleRecorder(VecEnvWrapper):
    def __init__(
        self,
        venv: VecEnv,
        video_path_prefix: str,
        video_step_interval: int = 1_000_000,
        max_video_length: int = 3600,
    ):
        super().__init__(venv)
        self.single_image_wrapper = VecSingleImageWrapper(venv)
        self.video_path_prefix = video_path_prefix
        self.video_step_interval = video_step_interval
        self.max_video_length = max_video_length
        self.total_steps = 0
        self.next_record_video_step = 0
        self.video_recorder = None
        self.recorded_frames = 0

    def step_wait(self) -> VecEnvStepReturn:
        obs, rew, dones, infos = self.venv.step_wait()
        self.total_steps += self.venv.num_envs
        # Using first env to record episodes
        if self.video_recorder:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if dones[0] and infos[0].get("episode"):
                episode_info = {
                    k: v.item() if hasattr(v, "item") else v
                    for k, v in infos[0]["episode"].items()
                }
                self.video_recorder.metadata["episode"] = episode_info
            if dones[0] or self.recorded_frames > self.max_video_length:
                self._close_video_recorder()
        elif dones[0] and self.total_steps >= self.next_record_video_step:
            self._start_video_recorder()
        return obs, rew, dones, infos

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        if self.video_recorder:
            self._close_video_recorder()
        elif (
            not self.video_recorder and self.total_steps >= self.next_record_video_step
        ):
            self._start_video_recorder()
        return obs

    def _start_video_recorder(self) -> None:
        self._close_video_recorder()

        video_path = f"{self.video_path_prefix}-{self.next_record_video_step}"
        self.video_recorder = VideoRecorder(
            self.single_image_wrapper,
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
