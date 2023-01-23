import gym
import os

from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from typing import Any, Callable, Dict, Optional

from shared.policy.policy import VEC_NORMALIZE_FILENAME


def make_env(
    env_id: str,
    seed: Optional[int],
    training: bool = True,
    render: bool = False,
    normalize_load_path: Optional[str] = None,
    n_envs: int = 1,
    frame_stack: int = 1,
    make_kwargs: Optional[Dict[str, Any]] = None,
    no_reward_timeout_steps: Optional[int] = None,
    vec_env_class: str = "dummy",
    normalize: bool = False,
    normalize_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    if "BulletEnv" in env_id:
        import pybullet_envs

    make_kwargs = make_kwargs if make_kwargs is not None else {}
    if "BulletEnv" in env_id and render:
        make_kwargs["render"] = True

    spec = gym.spec(env_id)

    def make(idx: int) -> Callable[[], gym.Env]:
        def _make() -> gym.Env:
            if "AtariEnv" in spec.entry_point:  # type: ignore
                from gym.wrappers.atari_preprocessing import AtariPreprocessing
                from gym.wrappers.frame_stack import FrameStack

                env = gym.make(env_id, **make_kwargs)
                env = AtariPreprocessing(env)
                env = FrameStack(env, frame_stack)
            elif "CarRacing" in env_id:
                from gym.wrappers.resize_observation import ResizeObservation
                from gym.wrappers.gray_scale_observation import GrayScaleObservation
                from gym.wrappers.frame_stack import FrameStack

                env = gym.make(env_id, verbose=0, **make_kwargs)
                env = ResizeObservation(env, (64, 64))
                env = GrayScaleObservation(env, keep_dim=False)
                env = FrameStack(env, frame_stack)
            else:
                env = gym.make(env_id, **make_kwargs)

            if no_reward_timeout_steps:
                from wrappers.no_reward_timeout import NoRewardTimeout

                env = NoRewardTimeout(env, no_reward_timeout_steps)

            if seed is not None:
                env.seed(seed + idx)
                env.action_space.seed(seed + idx)
                env.observation_space.seed(seed + idx)

            return env

        return _make

    VecEnvClass = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[vec_env_class]
    venv = VecEnvClass([make(i) for i in range(n_envs)])
    if normalize:
        if normalize_load_path:
            venv = VecNormalize.load(
                os.path.join(normalize_load_path, VEC_NORMALIZE_FILENAME), venv
            )
        else:
            venv = VecNormalize(venv, training=training, **(normalize_kwargs or {}))
        if not training:
            venv.norm_reward = False
    return venv
