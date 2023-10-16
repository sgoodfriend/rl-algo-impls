import gc
import hashlib
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
    def singleton(
        cls: Type[MicroRTSSocketEnvSelf], time_budget_ms: Optional[int] = None
    ) -> MicroRTSSocketEnvSelf:
        global _singleton
        if _singleton is None:
            _singleton = cls(time_budget_ms=time_budget_ms)
        return _singleton

    def __init__(self, time_budget_ms: Optional[int] = None):
        self.time_budget_ms = time_budget_ms if time_budget_ms else 100
        self._steps_since_reset = 0
        self._get_action_response_times = []
        self.listeners = []
        self._initialized = False

        self._logger = logging.getLogger("RTSServer")

        self.height = 0
        self.width = 0
        self._utt = None
        self._terrain = None
        self._terrain_md5 = None
        self._matrix_obs = None
        self._matrix_mask = None
        self.obs = None
        self.action_mask = None
        self._resources = None
        self._is_pre_game_analysis = False
        self._pre_game_analysis_expiration_ms = 0
        self._pre_game_analysis_folder: Optional[str] = None
        self._expected_step_ms = 0

        self.in_pipe = sys.stdin.buffer
        self.out_pipe = sys.stdout.buffer
        self._start()

    def step(self, action):
        if self.command == MessageType.PRE_GAME_ANALYSIS:
            self._send({"e": self._expected_step_ms, "a": action[0]})
        else:
            if self.command == MessageType.GET_ACTION:
                res_t = (time.perf_counter() - self._get_action_receive_time) * 1000
                self._get_action_response_times.append(res_t)
                if res_t >= self.time_budget_ms:
                    self._logger.warn(
                        f"Step: {self._steps_since_reset}: "
                        f"getAction response exceed threshold {int(res_t)}"
                    )
            self._send(action[0])
        return self._wait_for_obs()

    def reset(self, **kwargs):
        if not self._initialized:
            gc.disable()
            gc.collect()
            self._ack()
            self._initialized = True
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
        if not self._utt:
            self._logger.warn("utt unset. BAD IF NOT DEBUG")
            return {"unitTypes": [None] * 7}
        return self._utt

    @property
    def partial_obs(self) -> bool:
        return False

    def terrain(self, env_idx: int) -> np.ndarray:
        assert env_idx == 0
        assert self._terrain is not None
        return self._terrain

    def terrain_md5(self, env_idx: int) -> Optional[str]:
        assert env_idx == 0
        return self._terrain_md5

    def resources(self, env_idx: int) -> np.ndarray:
        assert env_idx == 0
        assert self._resources is not None
        return self._resources

    @property
    def is_pre_game_analysis(self) -> bool:
        return self._is_pre_game_analysis

    @property
    def pre_game_analysis_expiration_ms(self) -> int:
        return self._pre_game_analysis_expiration_ms

    @property
    def pre_game_analysis_folder(self) -> Optional[str]:
        return self._pre_game_analysis_folder

    def set_expected_step_ms(self, expected_step_ms: int) -> None:
        self._expected_step_ms = expected_step_ms

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
                    old_h = self.height
                    old_w = self.width
                    old_terrain = self._terrain
                    self.height, self.width = args[3]
                    self._terrain = np.frombuffer(args[4], dtype=np.int8).reshape(
                        (
                            self.height,
                            self.width,
                        )
                    )
                    hash_obj = hashlib.md5()
                    hash_obj.update(self._terrain.tobytes())
                    self._terrain_md5 = hash_obj.hexdigest()
                    map_change = (
                        self.height != old_h
                        or self.width != old_w
                        or old_terrain is None
                        or not np.array_equal(self._terrain, old_terrain)
                    )
                    if map_change:
                        for listener in self.listeners:
                            listener.map_change([old_h], [old_w], [old_terrain], [0])
                self.obs = [np.frombuffer(args[0], dtype=np.int8)]
                self.action_mask = [np.frombuffer(args[1], dtype=np.int8)]
                self._resources = np.frombuffer(args[2], dtype=np.int8)
                if self.command == MessageType.PRE_GAME_ANALYSIS:
                    self._is_pre_game_analysis = True
                    self._pre_game_analysis_expiration_ms = int(
                        time.perf_counter() * 1000
                    ) + int.from_bytes(args[5], byteorder="big")
                    self._pre_game_analysis_folder = args[6].decode("utf-8")
                    self._expected_step_ms = 0
                else:
                    self._is_pre_game_analysis = False
                    self._pre_game_analysis_expiration_ms = 0
                matrix_obs_idx = (
                    7 if self.command == MessageType.PRE_GAME_ANALYSIS else 5
                )
                matrix_mask_idx = matrix_obs_idx + 1
                if len(args) >= matrix_mask_idx:
                    self._matrix_obs = np.transpose(
                        np.array(json.loads(args[matrix_obs_idx].decode("utf-8"))),
                        (1, 2, 0),
                    )
                    self._matrix_mask = np.array(
                        json.loads(args[matrix_mask_idx].decode("utf-8")),
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
                        f"# Over {self.time_budget_ms}ms: {np.sum(res_times >= self.time_budget_ms)}"
                    )
                    self._get_action_response_times.clear()
                    self._steps_since_reset = 0

                self._logger.debug(f"Winner: {winner}")

                self.obs = None
                self.action_mask = None

                gc.disable()
                gc.collect()

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
        self.command, args = self._wait_for_message()
        assert self.command == MessageType.UTT
        self._utt = json.loads(args[0].decode("utf-8"))

    def debug_matrix_obs(self, env_idx: int) -> Optional[np.ndarray]:
        assert env_idx == 0
        return self._matrix_obs

    def debug_matrix_mask(self, env_idx: int) -> Optional[np.ndarray]:
        assert env_idx == 0
        return self._matrix_mask
