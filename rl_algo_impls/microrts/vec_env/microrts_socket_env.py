import json
import logging
import socket
import struct
import sys
import time
from typing import List, Tuple

import gym.spaces
import numpy as np

from rl_algo_impls.microrts.wrappers.microrts_space_transform import (
    MAX_HP,
    MAX_RESOURCES,
)

SERVER_PORT = 56241
IS_SERVER = True
MESSAGE_SIZE_BYTES = 8192
TIME_BUDGET_MS = 100


def set_connection_info(server_port: int, is_server: bool):
    global SERVER_PORT
    SERVER_PORT = server_port
    global IS_SERVER
    IS_SERVER = is_server


class MicroRTSSocketEnv:
    def __init__(self):
        self.partial_obs = False
        self._steps_since_reset = 0
        self._get_action_response_times = []

        self._logger = logging.getLogger("RTSServer")

        self._start()

    def step(self, action):
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

        self._wait_for_obs()
        return self.obs

    def _wait_for_obs(self):
        while True:
            command, args = self._wait_for_message()
            if command == "getAction":
                self._steps_since_reset += 1
                self._get_action_receive_time = time.perf_counter()
                self._handle_get_action(args)
                return self.obs, np.zeros(1), np.zeros(1), [{}]
            elif command == "gameOver":
                winner = json.loads(args[0])
                if winner == 0:
                    reward = 1
                elif winner == 1:
                    reward = -1
                else:
                    reward = 0
                self._ack()
                empty_obs = np.zeros_like(self.obs)
                self.obs = None
                return empty_obs, np.ones(1) * reward, np.ones(1), [{}]
            elif command == "utt":
                self.utt = json.loads(args[0])
                self._ack()

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

        self._connection.send(("%s\n" % data_string).encode("utf-8"))

    def _wait_for_message(self) -> Tuple[str, List[str]]:
        data_length = struct.unpack("!I", self._connection.recv(4))[0]
        d = bytearray()
        while len(d) < data_length:
            chunk = self._connection.recv(MESSAGE_SIZE_BYTES)
            if not chunk:
                break
            d.extend(chunk)
        environment_message = d.decode("utf-8")

        if environment_message[0] == "\n":
            return "ACK", []
        self._logger.debug("Message: %s" % environment_message)
        message_parts = environment_message.split("\n")
        self._logger.debug(message_parts[0])
        return message_parts[0], message_parts[1:]

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
                self._logger.critical(
                    "Bind failed. Error Code : " + str(msg[0]) + " Message " + msg[1]
                )
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

        high = np.dstack(
            [MAX_HP, MAX_RESOURCES, 3, len(self.utt["unitTypes"]) + 1, 6, 2]
            * self.height
            * self.width
        ).reshape((self.height, self.width, -1))
        if self.partial_obs:
            high = np.concatenate((high, np.full((self.height, self.width, 1), 2)))

        self.observation_space = gym.spaces.Box(
            low=0,
            high=high,  # type: ignore
            dtype=np.int32,
        )
        # This should be wrapped in a gym.spaces.Sequential, but Sequential doesn't
        # exist in gym 0.21
        self.action_space = gym.spaces.MultiDiscrete(
            [6, 4, 4, 4, 4, len(self.utt["unitTypes"]), 7 * 7]
        )

    def _handle_get_action(self, args: List[str]):
        self.obs = np.expand_dims(np.array(json.loads(args[0])), 0)
        self._action_mask = np.expand_dims(
            np.array(
                json.loads(args[1]),
            ),
            0,
        )
        map_data = json.loads(args[2])
        self.height = map_data["height"]
        self.width = map_data["width"]
