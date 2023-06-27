import gc
import json
import logging
import struct
import sys
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np

from rl_algo_impls.microrts.vec_env.microrts_interface import (
    MicroRTSInterface,
    MicroRTSInterfaceListener,
)

SERVER_PORT = 56242
IS_SERVER = True
MESSAGE_SIZE_BYTES = 8192
TIME_BUDGET_MS = 100


def set_connection_info(server_port: int, is_server: bool):
    global SERVER_PORT
    SERVER_PORT = server_port
    global IS_SERVER
    IS_SERVER = is_server


class MessageType(Enum):
    UTT = 0
    PRE_GAME_ANALYSIS = 1
    GET_ACTION = 2
    GAME_OVER = 3


message_types = {t.value: t for t in MessageType.__members__.values()}


MicroRTSSocketEnvSelf = TypeVar("MicroRTSSocketEnvSelf", bound="MicroRTSSocketEnv")
_singleton = None


class MicroRTSSocketEnv(MicroRTSInterface):
    @classmethod
    def singleton(cls: Type[MicroRTSSocketEnvSelf]) -> MicroRTSSocketEnvSelf:
        global _singleton
        if _singleton is None:
            _singleton = cls()
        return _singleton

    def __init__(self):
        self._steps_since_reset = 0
        self._get_action_response_times = []
        self.listeners = []

        self._logger = logging.getLogger("RTSServer")

        self.height = 0
        self.width = 0
        self._terrain = None
        self._matrix_obs = None
        self._matrix_mask = None
        self.obs = None
        self.action_mask = None
        self._resources = None

        self.in_pipe = sys.stdin.buffer
        self.out_pipe = sys.stdout.buffer
        self._start()

    def step(self, action):
        if self.command == MessageType.GET_ACTION:
            res_t = (time.perf_counter() - self._get_action_receive_time) * 1000
            self._get_action_response_times.append(res_t)
            if res_t >= TIME_BUDGET_MS:
                self._logger.warn(
                    f"Step: {self._steps_since_reset}: "
                    f"getAction response exceed threshold {int(res_t)}"
                )
        self._send(action[0])
        return self._wait_for_obs()

    def reset(self):
        if self.obs is not None:
            return self.obs, self.action_mask
        return self._wait_for_obs()[:2]

    @property
    def num_envs(self):
        return 1

    @property
    def heights(self) -> List[int]:
        return [self.height]

    @property
    def widths(self) -> List[int]:
        return [self.width]

    @property
    def utt(self) -> Dict[str, Any]:
        return self._utt

    @property
    def partial_obs(self) -> bool:
        return False

    def terrain(self, env_idx: int) -> np.ndarray:
        assert env_idx == 0
        assert self._terrain is not None
        return self._terrain

    def resources(self, env_idx: int) -> np.ndarray:
        assert env_idx == 0
        assert self._resources is not None
        return self._resources

    def close(self, **kwargs):
        pass

    def add_listener(self, listener: MicroRTSInterfaceListener) -> None:
        self.listeners.append(listener)

    def remove_listener(self, listener: MicroRTSInterfaceListener) -> None:
        self.listeners.remove(listener)

    def _wait_for_obs(self):
        while True:
            self.command, args = self._wait_for_message()
            if self.command == MessageType.UTT:
                self._utt = json.loads(args[0].decode("utf-8"))
                self._ack()
            elif self.command in {
                MessageType.PRE_GAME_ANALYSIS,
                MessageType.GET_ACTION,
            }:
                if self.command == MessageType.GET_ACTION:
                    self._steps_since_reset += 1
                    self._get_action_receive_time = time.perf_counter()
                if len(args) >= 5:
                    h, w = args[3]
                    map_size_change = self.height != h or self.width != w
                    self.height = h
                    self.width = w
                    self._terrain = np.frombuffer(args[4], dtype=np.int8).reshape(
                        (
                            self.height,
                            self.width,
                        )
                    )
                    if map_size_change:
                        for listener in self.listeners:
                            listener.map_size_change([h], [w], [0])
                self.obs = [np.frombuffer(args[0], dtype=np.int8)]
                self.action_mask = [np.frombuffer(args[1], dtype=np.int8)]
                self._resources = np.frombuffer(args[2], dtype=np.int8)
                if len(args) >= 7:
                    self._matrix_obs = np.transpose(
                        np.array(json.loads(args[5].decode("utf-8"))), (1, 2, 0)
                    )
                    self._matrix_mask = np.array(
                        json.loads(args[6].decode("utf-8")),
                    )
                return self.obs, self.action_mask, np.zeros(1), np.zeros(1), [{}]
            elif self.command == MessageType.GAME_OVER:
                winner = args[0][0]
                if winner == 0:
                    reward = 1
                elif winner == 1:
                    reward = -1
                else:
                    reward = 0

                if self._steps_since_reset:
                    res_times = np.array(self._get_action_response_times)
                    self._logger.info(
                        f"Steps: {self._steps_since_reset} - "
                        f"Average Response Time: {int(np.mean(res_times))}ms (std: {int(np.std(res_times))}ms) - "
                        f"Max Response Time: {int(np.max(res_times))}ms (Step: {np.argmax(res_times)}) - "
                        f"# Over {TIME_BUDGET_MS}ms: {np.sum(res_times >= TIME_BUDGET_MS)}"
                    )
                    self._get_action_response_times.clear()
                    self._steps_since_reset = 0

                self._logger.debug(f"Winner: {winner}")

                self.obs = None
                self.action_mask = None

                self._ack()

                self._wait_for_obs()
                return (
                    self.obs,
                    self.action_mask,
                    np.ones(1) * reward,
                    np.ones(1),
                    [{}],
                )
            else:
                raise ValueError(f"Unhandled command {self.command}")

    def _ack(self):
        self._send()

    def _send(self, data=None):
        if data is not None:
            data_string = json.dumps(data)
        else:
            data_string = ""

        if self.command == MessageType.PRE_GAME_ANALYSIS:
            gc.disable()
            gc.collect()
        self.out_pipe.write(("%s\n" % data_string).encode("utf-8"))
        self.out_pipe.flush()

    def _wait_for_message(self) -> Tuple[MessageType, List[bytearray]]:
        d = bytearray()
        while len(d) < 4:
            chunk = self.in_pipe.read(4)
            d.extend(chunk)
            if len(d) < 4:
                self._logger.debug(
                    f"Chunk ({chunk}) too small. Adding to buffer (now size {len(d)})"
                )

        sz = struct.unpack("!I", d[:4])[0]

        d = bytearray(d[4:])
        while len(d) < sz:
            chunk = self.in_pipe.read(sz)
            if not chunk:
                break
            d.extend(chunk)
            if len(d) < sz:
                self._logger.debug(
                    f"Chunk incomplete. Chunk size: {len(chunk)}, data size: {len(d)}, goal size: {sz}"
                )

        idx = 8
        t, n = struct.unpack("!II", d[:idx])
        part_sizes = struct.unpack("!" + "I" * n, d[idx : idx + 4 * n])
        idx += 4 * n
        parts = []
        for psz in part_sizes:
            parts.append(d[idx : idx + psz])
            idx += psz
        assert idx == sz

        message_type = message_types[t]

        self._logger.debug(f"Received {message_type}, size {sz}")
        return message_type, parts

    def _start(self):
        self._wait_for_obs()

    def debug_matrix_obs(self, env_idx: int) -> Optional[np.ndarray]:
        assert env_idx == 0
        return self._matrix_obs

    def debug_matrix_mask(self, env_idx: int) -> Optional[np.ndarray]:
        assert env_idx == 0
        return self._matrix_mask
