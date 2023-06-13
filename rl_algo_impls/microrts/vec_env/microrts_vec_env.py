import json
import logging
import os
import sys
import warnings
import xml.etree.ElementTree as ET
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypeVar

import gym
import gym.spaces
import jpype
import jpype.imports
import numpy as np
from jpype.imports import registerDomain
from jpype.types import JArray, JInt
from PIL import Image

from rl_algo_impls.microrts.wrappers.microrts_space_transform import (
    MAX_HP,
    MAX_RESOURCES,
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


class MicroRTSGridModeVecEnv:
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 150}
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
        reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0]),
        cycle_maps=[],
    ):
        self.num_selfplay_envs = num_selfplay_envs
        self.num_bot_envs = num_bot_envs
        self.num_envs = num_selfplay_envs + num_bot_envs
        assert self.num_bot_envs == len(
            ai2s
        ), "for each environment, a microrts ai should be provided"
        self.partial_obs = partial_obs
        self.max_steps = max_steps
        self.render_theme = render_theme
        self.frame_skip = frame_skip
        self.ai2s = ai2s
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

        self.map_actual_heights = []
        self.map_actual_widths = []
        self.height, self.width = (8, 8)
        for mp in map_paths:
            root = ET.parse(os.path.join(self.microrts_path, mp)).getroot()
            h = int(root.get("height"))
            w = int(root.get("width"))
            self.height = max(self.height, h)
            self.width = max(self.width, w)
            self.map_actual_heights.append(h)
            self.map_actual_widths.append(w)

        # launch the JVM
        if not jpype._jpype.isStarted():
            registerDomain("ts", alias="tests")
            registerDomain("ai")
            jars = [
                "rai.jar",
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

        # start microrts client
        from rts.units import UnitTypeTable

        self.real_utt = UnitTypeTable(UTT_VERSION_ORIGINAL_FINETUNED)
        from ai.reward import (
            AttackRewardFunction,
            ProduceBuildingRewardFunction,
            ProduceCombatUnitRewardFunction,
            ProduceWorkerRewardFunction,
            ResourceGatherRewardFunction,
            RewardFunctionInterface,
            ScoreRewardFunction,
            WinLossRewardFunction,
        )

        self.rfs = JArray(RewardFunctionInterface)(
            [
                WinLossRewardFunction(),
                ResourceGatherRewardFunction(),
                ProduceWorkerRewardFunction(),
                ProduceBuildingRewardFunction(),
                AttackRewardFunction(),
                ProduceCombatUnitRewardFunction(),
                ScoreRewardFunction(),
                # CloserToEnemyBaseRewardFunction(),
            ]
        )
        self.raw_names = [str(rf) for rf in self.rfs]
        self.start_client()

        high = np.dstack(self.obs_plane_space * self.height * self.width).reshape(
            (self.height, self.width, -1)
        )
        if partial_obs:
            high = np.concatenate((high, np.full((self.height, self.width, 1), 2)))

        self.observation_space = gym.spaces.Box(
            low=0,
            high=high,  # type: ignore
            dtype=np.int32,
        )
        # This should be wrapped in a gym.spaces.Sequential, but Sequential doesn't
        # exist in gym 0.21
        self.action_space = gym.spaces.MultiDiscrete(self.action_plane_space)

        self.delta_rewards = np.zeros(self.num_envs, dtype=np.float32)

    @property
    def obs_plane_space(self) -> Tuple[int, ...]:
        return (MAX_HP, MAX_RESOURCES, 3, len(self.utt["unitTypes"]) + 1, 6, 2)

    @property
    def action_plane_space(self) -> Tuple[int, ...]:
        return (6, 4, 4, 4, 4, len(self.utt["unitTypes"]), 7 * 7)

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
        self.utt = json.loads(str(self.render_client.sendUTT()))

    def _translate_observation(self, java_observation) -> np.ndarray:
        o = np.array(java_observation)
        map_h = o.shape[1]
        map_w = o.shape[2]
        if map_h == self.height and map_w == self.width:
            return o
        obs = np.zeros(
            (
                len(self.obs_plane_space),
                self.height,
                self.width,
            ),
            dtype=o.dtype,
        )
        obs[-1, :, :] = 1
        pad_h = (self.height - map_h) // 2
        pad_w = (self.width - map_w) // 2
        obs[:, pad_h : pad_h + map_h, pad_w : pad_w + map_w] = o
        return obs

    def _translate_observations(self, java_observations) -> np.ndarray:
        return np.array([self._translate_observation(o) for o in java_observations])

    def reset(self):
        self.delta_rewards = np.zeros_like(self.delta_rewards)

        responses = self.vec_client.reset([0] * self.num_envs)
        return self._translate_observations(responses.observation)

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
                winloss = rewards[self.raw_names.index("WinLossRewardFunction")]
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
        for idx, env_actions in enumerate(actions):
            map_h = self.map_actual_heights[idx]
            map_w = self.map_actual_widths[idx]
            if map_h == self.height and map_w == self.width:
                continue
            pad_h = (self.height - map_h) // 2
            pad_w = (self.width - map_w) // 2
            for a in env_actions:
                orig_h = a[0] // self.width
                orig_w = a[0] % self.width
                a[0] = (orig_h - pad_h) * map_w + orig_w - pad_w
        self.actions = JArray(JArray(JArray(JInt)))(
            [
                JArray(JArray(JInt))([JArray(JInt)(a) for a in actions_in_env])
                for actions_in_env in actions
            ]
        )

    def step_wait(self):
        responses = self.vec_client.gameStep(self.actions, [0] * self.num_envs)
        reward, done = np.array(responses.reward), np.array(responses.done)
        obs = self._translate_observations(responses.observation)
        infos = self._encode_info(reward, done[:, 0])
        # check if it is in evaluation, if not, then change maps
        if len(self.cycle_maps) > 0:
            # check if an environment is done, if done, reset the client, and replace the observation
            for done_idx, d in enumerate(done[:, 0]):
                # bot envs settings
                if done_idx < self.num_bot_envs:
                    if d:
                        self.vec_client.clients[done_idx].mapPath = next(self.next_map)
                        response = self.vec_client.clients[done_idx].reset(0)
                        obs[done_idx] = self._translate_observation(
                            response.observation
                        )
                # selfplay envs settings
                else:
                    if d and done_idx % 2 == 0:
                        done_idx -= self.num_bot_envs  # recalibrate the index
                        self.vec_client.selfPlayClients[done_idx // 2].mapPath = next(
                            self.next_map
                        )
                        self.vec_client.selfPlayClients[done_idx // 2].reset(0)
                        p0_response = self.vec_client.selfPlayClients[
                            done_idx // 2
                        ].getResponse(0)
                        p1_response = self.vec_client.selfPlayClients[
                            done_idx // 2
                        ].getResponse(1)
                        obs[done_idx] = self._translate_observation(
                            p0_response.observation
                        )
                        obs[done_idx + 1] = self._translate_observation(
                            p1_response.observation
                        )
        return obs, reward @ self.reward_weight, done[:, 0], infos

    def step(self, ac):
        self.step_async(ac)
        return self.step_wait()

    def getattr_depth_check(self, name, already_found):
        """
        Check if an attribute reference is being hidden in a recursive call to __getattr__
        :param name: (str) name of attribute to check for
        :param already_found: (bool) whether this attribute has already been found in a wrapper
        :return: (str or None) name of module whose attribute is being shadowed, if any.
        """
        if hasattr(self, name) and already_found:
            return "{0}.{1}".format(type(self).__module__, type(self).__name__)
        else:
            return None

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

    def close(self):
        if jpype._jpype.isStarted():
            self.vec_client.close()
            jpype.shutdownJVM()

    def get_action_mask(self):
        java_masks = self.vec_client.getMasks(0)
        masks = []
        for j_m in java_masks:
            m = np.array(j_m)
            map_h = m.shape[0]
            map_w = m.shape[1]
            if map_h == self.height and map_w == self.width:
                masks.append(m)
                continue
            new_m = np.zeros((self.height, self.width, m.shape[-1]), dtype=m.dtype)
            pad_h = (self.height - map_h) // 2
            pad_w = (self.width - map_w) // 2
            new_m[pad_h : pad_h + map_h, pad_w : pad_w + map_w] = m
            masks.append(new_m)
        return np.array(masks)

    @property
    def unwrapped(self: MicroRTSGridModeVecEnvSelf) -> MicroRTSGridModeVecEnvSelf:
        return self
