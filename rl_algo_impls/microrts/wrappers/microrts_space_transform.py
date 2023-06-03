import warnings
from typing import Any, Dict, List

import gym
import gym.spaces
import numpy as np

from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnvStepReturn,
    VectorableWrapper,
)

MAX_HP = 10
MAX_RESOURCES = 40


class MicroRTSSpaceTransform(VectorableWrapper):
    def __init__(
        self,
        env,
        utt: Dict[str, Any],
        height: int,
        width: int,
        partial_obs: bool = False,
    ) -> None:
        super().__init__(env)
        self.utt = utt
        self.partial_obs = partial_obs
        self.height = height
        self.width = width

        # computed properties
        # [hp, resources, resources_non_zero, num_planes_player(5),
        # num_planes_unit_type(z), num_planes_unit_action(6)]

        self.float_planes = [1 / MAX_HP, 1 / MAX_RESOURCES, None, None, None]
        self.bool_planes = [None, [1], None, None, None]
        self.one_hot_planes = [None, None, 3, len(self.utt["unitTypes"]) + 1, 6, 2]
        if partial_obs:
            self.one_hot_planes.append(2)
        self.n_dim = (
            len([fp for fp in self.float_planes if fp is not None])
            + sum([len(bp) for bp in self.bool_planes if bp is not None])
            + sum([ohp for ohp in self.one_hot_planes if ohp is not None])
        )

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.height, self.width, self.n_dim),
            dtype=np.float32,
        )
        self.single_observation_space = self.observation_space

        action_space_dims = [6, 4, 4, 4, 4, len(self.utt["unitTypes"]), 7 * 7]
        self.action_space = gym.spaces.MultiDiscrete(
            np.array([action_space_dims] * self.height * self.width).flatten().tolist()
        )
        self.single_action_space = self.action_space
        self.action_plane_space = gym.spaces.MultiDiscrete(action_space_dims)
        
        self.source_unit_idxs = np.tile(
            np.arange(self.height * self.width), (self.num_envs, 1)
        )
        self.source_unit_idxs = self.source_unit_idxs.reshape(
            (self.source_unit_idxs.shape + (1,))
        )

    def step(self, actions) -> VecEnvStepReturn:
        microrts_action = self._to_microrts_action(actions)
        microrts_obs, r, d, i = super().step(microrts_action)
        self._update_action_mask()

        obs = self._from_microrts_obs(microrts_obs)
        return obs, r, d, i

    def reset(self) -> np.ndarray:
        microrts_obs = self.env.reset()
        self._update_action_mask()
        return self._from_microrts_obs(microrts_obs)

    def _to_microrts_action(self, actions: np.ndarray) -> List[List[List[int]]]:
        actions = actions.reshape((self.num_envs, self.height * self.width, -1))
        actions = np.concatenate((self.source_unit_idxs, actions), 2)
        actions = actions[np.where(self.source_unit_mask == 1)]
        action_counts_per_env = self.source_unit_mask.sum(1)

        actions_per_env = []
        action_idx = 0
        for action_count in action_counts_per_env:
            actions_in_env = []
            for _ in range(action_count):
                actions_in_env.append(actions[action_idx].tolist())
                action_idx += 1
            actions_per_env.append(actions_in_env)
        return actions_per_env

    def _from_microrts_obs(self, microrts_obs) -> np.ndarray:
        return np.array([self._encode_obs(o) for o in microrts_obs])

    def _encode_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.reshape(len(obs), -1)
        obs_planes = np.zeros((self.height * self.width, self.n_dim), dtype=np.float32)

        plane_idx = 0
        for idx, fp in enumerate(self.float_planes):
            if fp is None:
                continue
            obs_planes[:, plane_idx] = obs[idx, :] * fp
            assert np.all(obs_planes[:, plane_idx] >= 0)
            if np.any(obs_planes[:, plane_idx] > 1):
                warnings.warn(
                    f"Found observations for plane_idx {plane_idx} above max ({np.max(obs[idx, :])})"
                )
            obs_planes[:, plane_idx] = obs_planes[:, plane_idx].clip(0, 1)
            plane_idx += 1
        for idx, bp in enumerate(self.bool_planes):
            if bp is None:
                continue
            for threshold in bp:
                obs_planes[:, plane_idx] = obs[idx, :] >= threshold
                plane_idx += 1
        for idx, ohp in enumerate(self.one_hot_planes):
            if ohp is None:
                continue
            obs_planes[:, plane_idx : plane_idx + ohp] = np.eye(ohp)[
                obs[idx, :].clip(0, ohp)
            ]
            plane_idx += ohp
        assert plane_idx == obs_planes.shape[-1]
        return obs_planes.reshape(self.height, self.width, -1)

    def get_action_mask(self) -> np.ndarray:
        return self._action_mask

    def _update_action_mask(self) -> None:
        action_mask = self.env.get_action_mask()
        self.source_unit_mask = action_mask[:, :, :, 0].reshape(self.num_envs, -1)
        self._action_mask = action_mask[:, :, :, 1:].reshape(
            self.num_envs, self.height * self.width, -1
        )
