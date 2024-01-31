import multiprocessing

import numpy as np
from gymnasium.error import AlreadyPendingCallError
from gymnasium.experimental.vector.async_vector_env import AsyncState, AsyncVectorEnv
from gymnasium.experimental.vector.sync_vector_env import SyncVectorEnv

from rl_algo_impls.wrappers.vector_wrapper import VecEnvMaskedResetReturn, VectorWrapper


class MaskResetWrapper(VectorWrapper):
    def masked_reset(self, env_mask: np.ndarray) -> VecEnvMaskedResetReturn:
        unwrapped = self.env.unwrapped
        if isinstance(unwrapped, SyncVectorEnv):
            return sync_masked_reset(unwrapped, env_mask)
        elif isinstance(unwrapped, AsyncVectorEnv):
            return async_masked_reset(unwrapped, env_mask)
        else:
            raise ValueError(f"Unsupported env type: {type(unwrapped)}")


def sync_masked_reset(
    vec_env: SyncVectorEnv, env_mask: np.ndarray
) -> VecEnvMaskedResetReturn:
    assert env_mask.dtype == bool
    assert env_mask.shape == (vec_env.num_envs,)
    assert env_mask.sum() > 0

    reset_envs = [
        env for env, should_reset in zip(vec_env.envs, env_mask) if should_reset
    ]
    obs = []
    infos = {}
    for idx, env in enumerate(reset_envs):
        ob, info = env.reset()
        obs.append(ob)
        vec_env._add_info(infos, info, idx)

    observations = np.stack(obs)

    vec_env._terminateds[env_mask] = False
    vec_env._truncateds[env_mask] = False
    vec_env.observations[env_mask] = observations

    return observations, None, infos


def async_masked_reset(
    vec_env: AsyncVectorEnv, env_mask: np.ndarray
) -> VecEnvMaskedResetReturn:
    assert env_mask.dtype == bool
    assert env_mask.shape == (vec_env.num_envs,)
    assert env_mask.sum() > 0

    vec_env._assert_is_running()

    if vec_env._state != AsyncState.DEFAULT:
        raise AlreadyPendingCallError(
            f"Calling `async_masked_reset` while in {vec_env._state.value} state to complete",
            str(vec_env._state.value),
        )

    pipes = []
    for pipe, should_reset in zip(vec_env.parent_pipes, env_mask):
        if should_reset:
            pipes.append(pipe)
            pipe.send(("reset", None))

    vec_env._state = AsyncState.WAITING_RESET

    if not vec_env._poll():
        vec_env._state = AsyncState.DEFAULT
        raise multiprocessing.TimeoutError(
            "The call to `async_masked_reset` has timed out."
        )

    results, successes = zip(*[pipe.recv() for pipe in pipes])
    vec_env._raise_if_errors(successes)
    vec_env._state = AsyncState.DEFAULT

    infos = {}
    obs, info_data = zip(*results)
    for idx, info in enumerate(info_data):
        vec_env._add_info(infos, info, idx)

    observations = np.stack(obs)
    if not vec_env.shared_memory:
        vec_env.observations[env_mask] = observations

    return observations, None, infos
