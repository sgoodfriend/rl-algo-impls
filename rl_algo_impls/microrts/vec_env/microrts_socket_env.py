import gc
import json
import logging
import socket
import struct
import sys
import time
from enum import Enum
from typing import List, Optional, Tuple

import gym.spaces
import numpy as np

from rl_algo_impls.microrts.wrappers.microrts_space_transform import (
    MAX_HP,
    MAX_RESOURCES,
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


class MicroRTSSocketEnv:
    def __init__(self):
        self.partial_obs = False
        self._steps_since_reset = 0
        self._get_action_response_times = []

        self._logger = logging.getLogger("RTSServer")

        self.terrain = None
        self.obs = None
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
            return self.obs
        self._wait_for_obs()
        return self.obs

    def _wait_for_obs(self):
        while True:
            self.command, args = self._wait_for_message()
            if self.command == MessageType.UTT:
                self.utt = json.loads(args[0].decode("utf-8"))
                self._ack()
            elif self.command in {
                MessageType.PRE_GAME_ANALYSIS,
                MessageType.GET_ACTION,
            }:
                if self.command == MessageType.GET_ACTION:
                    self._steps_since_reset += 1
                    self._get_action_receive_time = time.perf_counter()
                if len(args) >= 4:
                    self.height = args[2][0]
                    self.width = args[2][1]
                    self.terrain = np.frombuffer(args[3], dtype=np.int8).reshape(
                        (self.height, self.width)
                    )
                return self.parse_obs(*args[:2], *args[4:])
            elif self.command == MessageType.GAME_OVER:
                winner = args[0][0]
                if winner == 0:
                    reward = 1
                elif winner == 1:
                    reward = -1
                else:
                    reward = 0
                empty_obs = np.zeros_like(self.obs)
                self.obs = None

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
                self._ack()

                return empty_obs, np.ones(1) * reward, np.ones(1), [{}]
            else:
                raise ValueError(f"Unhandled command {self.command}")

    def get_action_mask(self) -> np.ndarray:
        return self._action_mask

    @property
    def unwrapped(self):
        return self

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
        self._connection.send(("%s\n" % data_string).encode("utf-8"))

    def _wait_for_message(self) -> Tuple[MessageType, List[bytearray]]:
        d = bytearray()
        while len(d) < 4:
            chunk = self._connection.recv(MESSAGE_SIZE_BYTES)
            d.extend(chunk)
            if len(d) < 4:
                self._logger.debug(
                    f"Chunk ({chunk}) too small. Adding to buffer (now size {len(d)})"
                )

        sz = struct.unpack("!I", d[:4])[0]

        d = bytearray(d[4:])
        while len(d) < sz:
            chunk = self._connection.recv(MESSAGE_SIZE_BYTES)
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
        """Start the MicroRTS server.

        :param start_microrts: whether to also start a MicroRTS instance along with the server. Be aware that, in order to spawn a java subprocess that will start MicroRTS, the MICRORTSPATH environment variable must be set, containing the path to the microRTS executable JAR. Defaults to False
        :type start_microrts: bool, optional
        :param properties_file: path to a properties file, which will be read by MicroRTS -f flag, defaults to None
        :type properties_file: str, optional
        """
        self._logger.info("Socket created")

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        addr = ("localhost", SERVER_PORT)

        if IS_SERVER:
            # Bind socket to local host and port
            try:
                s.bind(addr)
            except socket.error as msg:
                self._logger.critical(f"Bind failed. Error: {msg}")
                s.close()
                sys.exit()

            self._logger.info("Socket bind complete")

            # Start listening on socket
            s.listen(10)
            self._logger.info("Socket now listening")

            # now keep talking with the client
            self._connection, addr = s.accept()
        else:
            s.connect(addr)
            self._connection = s

        self._logger.info(self._connection)
        self._logger.info(addr)

        self._logger.info("Connected with " + addr[0] + ":" + str(addr[1]))

        self._wait_for_obs()

        high = np.dstack(self.obs_plane_space * self.height * self.width).reshape(
            (self.height, self.width, -1)
        )
        if self.partial_obs:
            high = np.concatenate((high, np.full((self.height, self.width, 1), 2)))

        self.observation_space = gym.spaces.Box(
            low=0,
            high=high,  # type: ignore
            dtype=np.int32,
        )
        # This should be wrapped in a gym.spaces.Sequential, but Sequential doesn't
        # exist in gym 0.21
        self.action_space = gym.spaces.MultiDiscrete(self.action_plane_space)

    @property
    def obs_plane_space(self) -> Tuple[int, ...]:
        return (MAX_HP, MAX_RESOURCES, 3, len(self.utt["unitTypes"]) + 1, 6, 2)

    @property
    def action_plane_space(self) -> Tuple[int, ...]:
        return (6, 4, 4, 4, 4, len(self.utt["unitTypes"]), 7 * 7)

    def parse_obs(
        self,
        obs_bytes: bytearray,
        mask_bytes: bytearray,
        obs_json_bytes: Optional[bytearray] = None,
        mask_json_bytes: Optional[bytearray] = None,
    ):
        obs_shape: Tuple[int, ...] = (len(self.obs_plane_space), self.height, self.width)  # type: ignore
        obs_array = np.frombuffer(obs_bytes, dtype=np.int8).reshape(
            (-1, obs_shape[0] + 1)
        )

        self.obs = np.zeros((1,) + obs_shape, dtype=np.int8)
        self.obs[0, -1] = self.terrain
        self.obs[0, :-1, obs_array[:, 0], obs_array[:, 1]] = obs_array[:, 2:]
        if obs_json_bytes:
            obs_reference = np.expand_dims(
                np.array(json.loads(obs_json_bytes.decode("utf-8"))), 0
            )
            obs_diff_indices = np.transpose(
                np.array(np.where(self.obs != obs_reference))
            )
            if len(obs_diff_indices):
                logging.error(f"Observation differences: {obs_diff_indices}")

        act_plane_dim = sum(self.action_plane_space)
        mask_array = np.frombuffer(mask_bytes, dtype=np.int8).reshape(
            -1, act_plane_dim + 2
        )
        self._action_mask = np.zeros(
            (1, self.height, self.width, act_plane_dim + 1),
            dtype=np.bool_,
        )
        self._action_mask[0, mask_array[:, 0], mask_array[:, 1], 0] = 1
        self._action_mask[0, mask_array[:, 0], mask_array[:, 1], 1:] = mask_array[:, 2:]

        if mask_json_bytes:
            mask_reference = np.expand_dims(
                np.array(
                    json.loads(mask_json_bytes.decode("utf-8")),
                ),
                0,
            )
            mask_diff_indices = np.transpose(
                np.array(np.where(self._action_mask != mask_reference))
            )
            if len(mask_diff_indices):
                logging.error(f"Mask differences: {mask_diff_indices}")

        return self.obs, np.zeros(1), np.zeros(1), [{}]
