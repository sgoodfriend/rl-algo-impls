import numpy as np

from gym.vector.sync_vector_env import SyncVectorEnv
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3.common.vec_env.base_vec_env import tile_images
from typing import Optional

from wrappers.vectorable_wrapper import (
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
            SyncVectorEnvRenderCompat(self.env),
            base_path=self.base_path,
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1

    def _close_video_recorder(self) -> None:
        if self.video_recorder:
            self.video_recorder.close()
        self.video_recorder = None


class SyncVectorEnvRenderCompat(VecotarableWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        base_env = self.env.unwrapped
        if isinstance(base_env, SyncVectorEnv):
            imgs = [env.render(mode="rgb_array") for env in base_env.envs]
            bigimg = tile_images(imgs)
            if mode == "humnan":
                import cv2

                cv2.imshow("vecenv", bigimg[:, :, ::-1])
                cv2.waitKey(1)
            elif mode == "rgb_array":
                return bigimg
            else:
                raise NotImplemented(f"Render mode {mode} is not supported")
        else:
            return self.env.render(mode=mode)
