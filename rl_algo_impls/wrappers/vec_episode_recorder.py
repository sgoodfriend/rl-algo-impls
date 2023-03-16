import numpy as np

from gym.wrappers.monitoring.video_recorder import VideoRecorder

from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecotarableWrapper,
    VecEnvObs,
    VecEnvStepReturn,
)


class VecEpisodeRecorder(VecotarableWrapper):
    def __init__(self, env, base_path: str, max_video_length: int = 3600):
        super().__init__(env)
        self.base_path = base_path
        self.max_video_length = max_video_length
        self.video_recorder = None
        self.recorded_frames = 0

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        obs, rew, dones, infos = self.env.step(actions)
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
        return obs, rew, dones, infos

    def reset(self) -> VecEnvObs:
        obs = self.env.reset()
        self._start_video_recorder()
        return obs

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
        self.video_recorder = None
