import atexit
import json
import logging
import os
import sys
import warnings
import xml.etree.ElementTree as ET
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

import jpype
import jpype.imports
import numpy as np
from jpype.imports import registerDomain
from jpype.types import JArray, JInt
from PIL import Image

from rl_algo_impls.microrts.vec_env.microrts_interface import (
    ByteArray,
    MicroRTSInterface,
    MicroRTSInterfaceListener,
)

MICRORTS_MAC_OS_RENDER_MESSAGE = """
gym-microrts render is not available on MacOS. See https://github.com/jpype-project/jpype/issues/906

It is however possible to record the videos via `env.render(mode='rgb_array')`. 
See https://github.com/vwxyzjn/gym-microrts/blob/b46c0815efd60ae959b70c14659efb95ef16ffb0/hello_world_record_video.py
as an example.
"""

MicroRTSGridModeVecEnvSelf = TypeVar(
    "MicroRTSGridModeVecEnvSelf", bound="MicroRTSGridModeVecEnv"
)

UTT_VERSION_ORIGINAL_FINETUNED = 2


class MicroRTSGridModeVecEnv(MicroRTSInterface):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 150}
    DEBUG_VERIFY = False
    """
    [[0]x_coordinate*y_coordinate(x*y), [1]a_t(6), [2]p_move(4), [3]p_harvest(4), 
    [4]p_return(4), [5]p_produce_direction(4), [6]p_produce_unit_type(z), 
    [7]x_coordinate*y_coordinate(x*y)]
    Create a baselines VecEnv environment from a gym3 environment.
    :param env: gym3 environment to adapt
    """

    def __init__(
        self,
        num_selfplay_envs,
        num_bot_envs,
        partial_obs=False,
        max_steps=2000,
        render_theme=2,
        frame_skip=0,
        ai2s=[],
        map_paths=["maps/10x10/basesTwoWorkers10x10.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0, 5.25, 6.0, 0]),
        cycle_maps=[],
        bot_envs_alternate_player: bool = False,
    ):
        self.num_selfplay_envs = num_selfplay_envs
        self.num_bot_envs = num_bot_envs
        self._num_envs = num_selfplay_envs + num_bot_envs
        assert self.num_bot_envs == len(
            ai2s
        ), "for each environment, a microrts ai should be provided"
        self._partial_obs = partial_obs
        self.max_steps = max_steps
        self.render_theme = render_theme
        self.frame_skip = frame_skip
        self.ai2s = ai2s
        self.players = [i % 2 for i in range(self._num_envs)]
        self.bot_envs_alternate_player = bot_envs_alternate_player
        self.map_paths = map_paths
        if len(map_paths) == 1:
            self.map_paths = [map_paths[0] for _ in range(self.num_envs)]
        else:
            assert (
                len(map_paths) == self.num_envs
            ), "if multiple maps are provided, they should be provided for each environment"
        self.reward_weight = reward_weight

        self.microrts_path = os.path.join(Path(__file__).parent.parent, "java")

        # prepare training maps
        self.cycle_maps = list(
            map(lambda i: os.path.join(self.microrts_path, i), cycle_maps)
        )
        self.next_map = cycle(self.cycle_maps)

        self._heights = []
        self._widths = []
        for mp in self.map_paths:
            root = ET.parse(os.path.join(self.microrts_path, mp)).getroot()
            h = int(root.get("height"))
            w = int(root.get("width"))
            self._heights.append(h)
            self._widths.append(w)

        # launch the JVM
        if not jpype._jpype.isStarted():
            registerDomain("ts", alias="tests")
            registerDomain("ai")
            jars = [
                "RAISocketAI.jar",
                "lib/microrts.jar",
                "lib/bots/Coac.jar",
                "lib/bots/Droplet.jar",
                "lib/bots/GRojoA3N.jar",
                "lib/bots/Izanagi.jar",
                "lib/bots/MixedBot.jar",
                "lib/bots/TiamatBot.jar",
                "lib/bots/UMSBot.jar",
                "lib/bots/mayariBot.jar",  # "MindSeal.jar"
            ]
            for jar in jars:
                jpype.addClassPath(os.path.join(self.microrts_path, jar))
            jpype.startJVM(convertStrings=False)
            atexit.register(jpype.shutdownJVM)

        # start microrts client
        from rts.units import UnitTypeTable

        self.real_utt = UnitTypeTable(UTT_VERSION_ORIGINAL_FINETUNED)
        from ai.reward import (
            AttackRewardFunction,
            ProduceBuildingRewardFunction,
            ProduceHeavyUnitRewardFunction,
            ProduceLightUnitRewardFunction,
            ProduceRangedUnitRewardFunction,
            ProduceWorkerRewardFunction,
            RAIWinLossRewardFunction,
            ResourceGatherRewardFunction,
            RewardFunctionInterface,
            ScoreRewardFunction,
        )

        self.rfs = JArray(RewardFunctionInterface)(
            [
                RAIWinLossRewardFunction(),
                ResourceGatherRewardFunction(),
                ProduceWorkerRewardFunction(),
                ProduceBuildingRewardFunction(),
                AttackRewardFunction(),
                ProduceLightUnitRewardFunction(),
                ProduceRangedUnitRewardFunction(),
                ProduceHeavyUnitRewardFunction(),
                ScoreRewardFunction(),
                # CloserToEnemyBaseRewardFunction(),
            ]
        )
        self.raw_names = [str(rf) for rf in self.rfs]
        self.start_client()

        self.delta_rewards = np.zeros(self.num_envs, dtype=np.float32)

    def start_client(self):
        from ai.core import AI
        from ts import RAIGridnetVecClient as Client

        self.vec_client = Client(
            self.num_selfplay_envs,
            self.num_bot_envs,
            self.max_steps,
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_paths,
            JArray(AI)([ai2(self.real_utt) for ai2 in self.ai2s]),
            self.real_utt,
            self.partial_obs,
        )
        self.render_client = (
            self.vec_client.selfPlayClients[0]
            if len(self.vec_client.selfPlayClients) > 0
            else self.vec_client.clients[0]
        )
        # get the unit type table
        self._utt = json.loads(str(self.render_client.sendUTT()))

    def reset(self):
        self.delta_rewards = np.zeros_like(self.delta_rewards)

        responses = self.vec_client.reset(self.players)
        self._resources = to_byte_array_list(responses.resources)
        self._terrain = to_byte_array_list(responses.terrain)
        return (
            to_byte_array_list(responses.observation),
            to_byte_array_list(responses.mask),
        )

    def _encode_info(
        self, batch_rewards: np.ndarray, done: np.ndarray
    ) -> List[Dict[str, Any]]:
        infos = []
        for idx, (rewards, d) in enumerate(zip(batch_rewards, done)):
            score_reward = rewards[self.raw_names.index("ScoreRewardFunction")]
            delta_reward = score_reward - self.delta_rewards[idx]
            info = {
                "raw_rewards": rewards,
                "score_reward": {
                    "reward": score_reward,
                    "delta_reward": delta_reward,
                },
            }
            if d:
                winloss = rewards[self.raw_names.index("RAIWinLossRewardFunction")]
                if np.sign(score_reward) != np.sign(winloss) and np.sign(winloss) != 0:
                    logging.warn(
                        f"score_reward {score_reward} must be same sign as winloss {winloss}"
                    )
                info["results"] = {
                    "score_reward": score_reward,
                    "WinLoss": winloss,
                    "win": int(winloss == 1),
                    "draw": int(winloss == 0),
                    "loss": int(winloss == -1),
                }
            infos.append(info)
            self.delta_rewards[idx] = score_reward if not d else 0
        return infos

    def step_async(self, actions):
        self.actions = JArray(JArray(JArray(JInt)))(
            [
                JArray(JArray(JInt))([JArray(JInt)(a) for a in actions_in_env])
                for actions_in_env in actions
            ]
        )

    def step_wait(self):
        responses = self.vec_client.gameStep(self.actions, self.players)
        reward, done = np.array(responses.reward), np.array(responses.done)
        obs = to_byte_array_list(responses.observation)
        mask = to_byte_array_list(responses.mask)
        self._resources = to_byte_array_list(responses.resources)
        infos = self._encode_info(reward, done[:, 0])
        # check if it is in evaluation, if not, then change maps
        if len(self.cycle_maps) > 0:
            # check if an environment is done, if done, reset the client, and replace the observation
            for done_idx, d in enumerate(done[:, 0]):
                if not d:
                    continue
                if done_idx < self.num_selfplay_envs:
                    if done_idx % 2 == 1:
                        continue
                    self.vec_client.selfPlayClients[done_idx // 2].mapPath = next(
                        self.next_map
                    )
                    self.vec_client.selfPlayClients[done_idx // 2].reset(
                        self.players[done_idx]
                    )
                    p0_response = self.vec_client.selfPlayClients[
                        done_idx // 2
                    ].getResponse(0)
                    p1_response = self.vec_client.selfPlayClients[
                        done_idx // 2
                    ].getResponse(1)
                    obs[done_idx] = np.array(p0_response.observation)
                    obs[done_idx + 1] = np.array(p1_response.observation)
                    mask[done_idx] = np.array(p0_response.mask)
                    mask[done_idx + 1] = np.array(p1_response.mask)
                    self._terrain[done_idx] = np.array(p0_response.terrain)
                    self._terrain[done_idx + 1] = np.array(p1_response.terrain)
                    self._resources[done_idx] = np.array(p0_response.resources)
                    self._resources[done_idx + 1] = np.array(p1_response.resources)
                else:
                    env_idx = done_idx - self.num_selfplay_envs
                    self.vec_client.clients[env_idx].mapPath = next(self.next_map)
                    if self.bot_envs_alternate_player:
                        self.players[done_idx] = (self.players[done_idx] + 1) % 2
                    response = self.vec_client.clients[env_idx].reset(
                        self.players[done_idx]
                    )
                    obs[done_idx] = np.array(response.observation)
                    mask[done_idx] = np.array(response.mask)
                    self._terrain[done_idx] = np.array(response.terrain)
                    self._resources[done_idx] = np.array(response.resources)
        else:
            if self.bot_envs_alternate_player:
                for done_idx, d in enumerate(done[:, 0]):
                    if not d:
                        continue
                    if done_idx < self.num_selfplay_envs:
                        continue
                    self.players[done_idx] = (self.players[done_idx] + 1) % 2
                    env_idx = done_idx - self.num_selfplay_envs
                    response = self.vec_client.clients[env_idx].reset(
                        self.players[done_idx]
                    )
                    obs[done_idx] = np.array(response.observation)
                    mask[done_idx] = np.array(response.mask)
                    self._terrain[done_idx] = np.array(response.terrain)
                    self._resources[done_idx] = np.array(response.resources)
        return obs, mask, reward @ self.reward_weight, done[:, 0], infos

    def step(self, ac):
        self.step_async(ac)
        return self.step_wait()

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def heights(self) -> List[int]:
        return self._heights

    @property
    def widths(self) -> List[int]:
        return self._widths

    @property
    def utt(self) -> Dict[str, Any]:
        return self._utt

    @property
    def partial_obs(self) -> bool:
        return self._partial_obs

    def terrain(self, env_idx: int) -> np.ndarray:
        return self._terrain[env_idx].reshape(
            (self._heights[env_idx], self._widths[env_idx])
        )

    def terrain_md5(self, env_idx: int) -> str:
        import hashlib

        hash_obj = hashlib.md5()
        hash_obj.update(self.terrain(env_idx).tobytes())
        return hash_obj.hexdigest()

    def resources(self, env_idx: int) -> np.ndarray:
        return self._resources[env_idx]

    def render(self, mode="human"):
        if mode == "human":
            self.render_client.render(False)
            # give warning on macos because the render is not available
            if sys.platform == "darwin":
                warnings.warn(MICRORTS_MAC_OS_RENDER_MESSAGE)
        elif mode == "rgb_array":
            bytes_array = np.array(self.render_client.render(True))
            image = Image.frombytes("RGB", (640, 640), bytes_array)
            return np.array(image)[:, :, ::-1]

    def close(self, **kwargs):
        if jpype._jpype.isStarted():
            self.vec_client.close()

    def add_listener(self, listener: MicroRTSInterfaceListener) -> None:
        pass

    def remove_listener(self, listener: MicroRTSInterfaceListener) -> None:
        pass

    def debug_matrix_obs(self, env_idx: int) -> Optional[np.ndarray]:
        if not self.DEBUG_VERIFY:
            return None
        from ai.rai import GameStateWrapper

        if env_idx < self.num_selfplay_envs:
            gsw = GameStateWrapper(self.vec_client.selfPlayClients[env_idx // 2].gs)
            player_id = env_idx % 2
        else:
            gsw = GameStateWrapper(
                self.vec_client.clients[env_idx - self.num_selfplay_envs].gs
            )
            player_id = self.players[env_idx]
        return np.transpose(np.array(gsw.getVectorObservation(player_id)), (1, 2, 0))

    def debug_matrix_mask(self, env_idx: int) -> Optional[np.ndarray]:
        if not self.DEBUG_VERIFY:
            return None

        from ai.rai import GameStateWrapper

        if env_idx < self.num_selfplay_envs:
            gsw = GameStateWrapper(self.vec_client.selfPlayClients[env_idx // 2].gs)
            player_id = env_idx % 2
        else:
            gsw = GameStateWrapper(
                self.vec_client.clients[env_idx - self.num_selfplay_envs].gs
            )
            player_id = self.players[env_idx]
        return np.array(gsw.getMasks(player_id))


def to_byte_array_list(java_array) -> List[ByteArray]:
    return [np.array(a) for a in java_array]
