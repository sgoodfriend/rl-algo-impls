import multiprocessing as mp
import sys
import time
from copy import deepcopy
from ctypes import c_bool
from enum import Enum
from typing import Any, List, Optional, Union

import numpy as np
from gym import logger
from gym.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
    NoAsyncCallError,
)
from gym.vector.utils import (
    CloudpickleWrapper,
    clear_mpi_env_vars,
    concatenate,
    create_empty_array,
    create_shared_memory,
    read_from_shared_memory,
    write_to_shared_memory,
)
from gym.vector.vector_env import VectorEnv

from rl_algo_impls.wrappers.lux_env_gridnet import LuxEnvGridnet, LuxRewardWeights

__all__ = ["LuxAsyncVectorEnv"]


class AsyncState(Enum):
    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"
    WAITING_CALL = "call"


class LuxAsyncVectorEnv(VectorEnv):
    """Vectorized environment that runs multiple environments in parallel. It
    uses `multiprocessing` processes, and pipes for communication.

    Parameters
    ----------
    env_fns : iterable of callable
        Functions that create the environments.

    observation_space : `gym.spaces.Space` instance, optional
        Observation space of a single environment. If `None`, then the
        observation space of the first environment is taken.

    action_space : `gym.spaces.Space` instance, optional
        Action space of a single environment. If `None`, then the action space
        of the first environment is taken.

    shared_memory : bool (default: `True`)
        If `True`, then the observations from the worker processes are
        communicated back through shared variables. This can improve the
        efficiency if the observations are large (e.g. images).

    copy : bool (default: `True`)
        If `True`, then the `reset` and `step` methods return a copy of the
        observations.

    context : str, optional
        Context for multiprocessing. If `None`, then the default context is used.
        Only available in Python 3.

    daemon : bool (default: `True`)
        If `True`, then subprocesses have `daemon` flag turned on; that is, they
        will quit if the head process quits. However, `daemon=True` prevents
        subprocesses to spawn children, so for some environments you may want
        to have it set to `False`

    worker : function, optional
        WARNING - advanced mode option! If set, then use that worker in a subprocess
        instead of a default one. Can be useful to override some inner vector env
        logic, for instance, how resets on done are handled. Provides high
        degree of flexibility and a high chance to shoot yourself in the foot; thus,
        if you are writing your own worker, it is recommended to start from the code
        for `_worker` (or `_worker_shared_memory`) method below, and add changes
    """

    def __init__(
        self,
        env_fns,
        observation_space=None,
        action_space=None,
        shared_memory=True,
        copy=True,
        context=None,
        daemon=True,
        worker=None,
    ):
        ctx = mp.get_context(context)
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy
        dummy_env = env_fns[0]()
        assert isinstance(dummy_env, LuxEnvGridnet)
        self.metadata = dummy_env.metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or dummy_env.single_observation_space
            action_space = action_space or dummy_env.single_action_space
        self._reward_weights = dummy_env.reward_weights
        super(LuxAsyncVectorEnv, self).__init__(
            num_envs=len(env_fns) * 2,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.action_plane_space = dummy_env.action_plane_space

        if self.shared_memory:
            try:
                _obs_buffer = ctx.Array(
                    self.single_observation_space.dtype.char,
                    self.num_envs * int(np.prod(self.single_observation_space.shape)),
                )
                self.observations = np.frombuffer(
                    _obs_buffer.get_obj(), dtype=self.single_observation_space.dtype
                ).reshape((self.num_envs,) + self.single_observation_space.shape)
                _action_masks_buffer = ctx.Array(
                    c_bool,
                    self.num_envs * int(np.prod(dummy_env.action_mask_shape)),
                )
                self.action_masks = np.frombuffer(
                    _action_masks_buffer.get_obj(), dtype=np.bool_
                ).reshape((self.num_envs,) + dummy_env.action_mask_shape)
            except CustomSpaceError:
                raise ValueError(
                    "Using `shared_memory=True` in `AsyncVectorEnv` "
                    "is incompatible with non-standard Gym observation spaces "
                    "(i.e. custom spaces inheriting from `gym.Space`), and is "
                    "only compatible with default Gym spaces (e.g. `Box`, "
                    "`Tuple`, `Dict`) for batching. Set `shared_memory=False` "
                    "if you use custom observation spaces."
                )
        else:
            _obs_buffer = None
            self.observations = create_empty_array(
                self.single_observation_space, n=self.num_envs, fn=np.zeros
            )
            self.observations = np.zeros(
                (self.num_envs,) + self.single_observation_space.shape,
                dtype=self.single_observation_space.dtype,
            )
            _action_masks_buffer = None
            self.action_masks = np.full(
                (self.num_envs,) + dummy_env.action_mask_shape,
                False,
                dtype=np.bool_,
            )

        dummy_env.close()
        del dummy_env

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = _worker_shared_memory if self.shared_memory else _worker
        target = worker or target
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name="Worker<{0}>-{1}".format(type(self).__name__, idx),
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        _obs_buffer,
                        self.error_queue,
                        _action_masks_buffer,
                    ),
                )

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()

        self._state = AsyncState.DEFAULT
        self._check_observation_spaces()

    def seed(self, seeds=None):
        self._assert_is_running()
        if seeds is None:
            seeds = [None for _ in range(self.num_envs)]
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        assert len(seeds) == self.num_envs

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `seed` while waiting "
                "for a pending call to `{0}` to complete.".format(self._state.value),
                self._state.value,
            )

        for pipe, seed in zip(self.parent_pipes, seeds):
            pipe.send(("seed", seed))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

    def reset_async(self):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `reset_async` while waiting "
                "for a pending call to `{0}` to complete".format(self._state.value),
                self._state.value,
            )

        for pipe in self.parent_pipes:
            pipe.send(("reset", None))
        self._state = AsyncState.WAITING_RESET

    def reset_wait(self, timeout=None):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `reset_wait` times out. If
            `None`, the call to `reset_wait` never times out.

        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError(
                "Calling `reset_wait` without any prior " "call to `reset_async`.",
                AsyncState.WAITING_RESET.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                "The call to `reset_wait` has timed out after "
                "{0} second{1}.".format(timeout, "s" if timeout > 1 else "")
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        if not self.shared_memory:
            obs, action_masks = zip(*results)
            self.observations = np.concatenate(obs)
            self.action_masks = np.concatenate(action_masks)

        return deepcopy(self.observations) if self.copy else self.observations

    def step_async(self, actions):
        """
        Parameters
        ----------
        actions : iterable of samples from `action_space`
            List of actions.
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `step_async` while waiting "
                "for a pending call to `{0}` to complete.".format(self._state.value),
                self._state.value,
            )

        paired_actions = np.array(np.split(actions, len(actions) // 2, axis=0))
        for pipe, action in zip(self.parent_pipes, paired_actions):
            pipe.send(("step", action))
        self._state = AsyncState.WAITING_STEP

    def step_wait(self, timeout=None):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `step_wait` times out. If
            `None`, the call to `step_wait` never times out.

        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.

        rewards : `np.ndarray` instance (dtype `np.float_`)
            A vector of rewards from the vectorized environment.

        dones : `np.ndarray` instance (dtype `np.bool_`)
            A vector whose entries indicate whether the episode has ended.

        infos : list of dict
            A list of auxiliary diagnostic information.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `step_wait` without any prior call " "to `step_async`.",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                "The call to `step_wait` has timed out after "
                "{0} second{1}.".format(timeout, "s" if timeout > 1 else "")
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT
        observations_list, rewards, dones, infos, action_masks = zip(*results)

        if not self.shared_memory:
            self.observations = np.concatenate(observations_list)
            self.action_masks = np.concatenate(action_masks)

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.concatenate(rewards),
            np.concatenate(dones, dtype=np.bool_),
            [info for pair in infos for info in pair],
        )

    def call_async(self, name: str, *args, **kwargs):
        """Calls the method with name asynchronously and apply args and kwargs to the method.

        Args:
            name: Name of the method or property to call.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: Calling `call_async` while waiting for a pending call to complete
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `call_async` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for pipe in self.parent_pipes:
            pipe.send(("_call", (name, args, kwargs)))
        self._state = AsyncState.WAITING_CALL

    def call_wait(self, timeout: Optional[Union[int, float]] = None) -> list:
        """Calls all parent pipes and waits for the results.

        Args:
            timeout: Number of seconds before the call to `step_wait` times out.
                If `None` (default), the call to `step_wait` never times out.

        Returns:
            List of the results of the individual calls to the method or property for each environment.

        Raises:
            NoAsyncCallError: Calling `call_wait` without any prior call to `call_async`.
            TimeoutError: The call to `call_wait` has timed out after timeout second(s).
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_wait` without any prior call to `call_async`.",
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `call_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return results

    def call(self, name: str, *args, **kwargs) -> List[Any]:
        self.call_async(name, *args, **kwargs)
        return self.call_wait()

    def get_attr(self, name: str):
        return self.call(name)

    def set_attr(self, name: str, values: Union[list, tuple, object]):
        """Sets an attribute of the sub-environments.

        Args:
            name: Name of the property to be set in each individual environment.
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
            AlreadyPendingCallError: Calling `set_attr` while waiting for a pending call to complete.
        """
        self._assert_is_running()
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `set_attr` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for pipe, value in zip(self.parent_pipes, values):
            pipe.send(("_setattr", (name, value)))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

    @property
    def reward_weights(self) -> LuxRewardWeights:
        assert self._reward_weights is not None
        return self._reward_weights

    @reward_weights.setter
    def reward_weights(self, reward_weights: LuxRewardWeights) -> None:
        self._reward_weights = reward_weights
        self.set_attr("reward_weights", reward_weights)

    def get_action_mask(self) -> np.ndarray:
        return self.action_masks

    def close_extras(self, timeout=None, terminate=False):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `close` times out. If `None`,
            the call to `close` never times out. If the call to `close` times
            out, then all processes are terminated.

        terminate : bool (default: `False`)
            If `True`, then the `close` operation is forced and all processes
            are terminated.
        """
        timeout = 0 if terminate else timeout
        try:
            if self._state != AsyncState.DEFAULT:
                logger.warn(
                    "Calling `close` while waiting for a pending "
                    "call to `{0}` to complete.".format(self._state.value)
                )
                function = getattr(self, "{0}_wait".format(self._state.value))
                function(timeout)
        except mp.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(("close", None))
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()

        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()

    def _poll(self, timeout=None):
        self._assert_is_running()
        if timeout is None:
            return True
        end_time = time.perf_counter() + timeout
        delta = None
        for pipe in self.parent_pipes:
            delta = max(end_time - time.perf_counter(), 0)
            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True

    def _check_observation_spaces(self):
        self._assert_is_running()
        for pipe in self.parent_pipes:
            pipe.send(("_check_observation_space", self.single_observation_space))
        same_spaces, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        if not all(same_spaces):
            raise RuntimeError(
                "Some environments have an observation space "
                "different from `{0}`. In order to batch observations, the "
                "observation spaces from all environments must be "
                "equal.".format(self.single_observation_space)
            )

    def _assert_is_running(self):
        if self.closed:
            raise ClosedEnvironmentError(
                "Trying to operate on `{0}`, after a "
                "call to `close()`.".format(type(self).__name__)
            )

    def _raise_if_errors(self, successes):
        if all(successes):
            return

        num_errors = self.num_envs - sum(successes)
        assert num_errors > 0
        for _ in range(num_errors):
            index, exctype, value = self.error_queue.get()
            logger.error(
                "Received the following error from Worker-{0}: "
                "{1}: {2}".format(index, exctype.__name__, value)
            )
            logger.error("Shutting down Worker-{0}.".format(index))
            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

        logger.error("Raising the last exception back to the main process.")
        raise exctype(value)


def _worker(
    index, env_fn, pipe, parent_pipe, shared_memory, error_queue, action_masks_buffer
):
    assert shared_memory is None
    assert action_masks_buffer is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation = env.reset()
                action_mask = env.get_action_mask()
                pipe.send(((observation, action_mask), True))
            elif command == "step":
                observation, reward, done, info = env.step(data)
                if all(done):
                    observation = env.reset()
                action_mask = env.get_action_mask()
                pipe.send(((observation, reward, done, info, action_mask), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_check_observation_space":
                pipe.send((data == env.single_observation_space, True))
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            else:
                raise RuntimeError(
                    "Received unknown command `{0}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, "
                    "`_check_observation_space`}.".format(command)
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def np_array_to_shared_memory(index: int, shared_memory, np_array: np.ndarray) -> None:
    size = int(np.prod(np_array.shape))
    destination = np.frombuffer(shared_memory.get_obj(), dtype=np_array.dtype)
    np.copyto(
        destination[index * size : (index + 1) * size],
        np.asarray(np_array, dtype=np_array.dtype).flatten(),
    )


def _worker_shared_memory(
    index, env_fn, pipe, parent_pipe, shared_memory, error_queue, action_masks_buffer
):
    assert shared_memory is not None
    assert action_masks_buffer is not None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation = env.reset()
                np_array_to_shared_memory(index, shared_memory, observation)
                action_mask = env.get_action_mask()
                np_array_to_shared_memory(index, action_masks_buffer, action_mask)
                pipe.send(((None, None), True))
            elif command == "step":
                observation, reward, done, info = env.step(data)
                if all(done):
                    observation = env.reset()
                np_array_to_shared_memory(index, shared_memory, observation)
                action_mask = env.get_action_mask()
                np_array_to_shared_memory(index, action_masks_buffer, action_mask)
                pipe.send(((None, reward, done, info, None), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_observation_space":
                pipe.send((data == env.single_observation_space, True))
            else:
                raise RuntimeError(
                    "Received unknown command `{0}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, "
                    "`_check_observation_space`}.".format(command)
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
