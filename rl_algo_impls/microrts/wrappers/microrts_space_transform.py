import logging
import warnings
from typing import Any, Dict, List

import gym
import gym.spaces
import gym.vector
import numpy as np

from rl_algo_impls.microrts.vec_env.microrts_interface import (
    ByteArray,
    MicroRTSInterface,
)
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnvStepReturn

MAX_HP = 10
MAX_RESOURCES = 40
ACTION_TYPE_TO_ACTION_INDEXES = {1: {1}, 2: {2}, 3: {3}, 4: {4, 5}, 5: {6}}


class MicroRTSSpaceTransform(gym.vector.VectorEnv):
    def __init__(
        self,
        interface: MicroRTSInterface,
    ) -> None:
        self.interface = interface
        # Set height and width to next factor of 4 if not factor of 4 already
        next_factor_of_4 = lambda n: n + 4 - n % 4 if n % 4 else n
        height = max(next_factor_of_4(np.max(self.interface.heights)), 16)
        width = max(next_factor_of_4(np.max(self.interface.widths)), 16)
        assert height % 4 == 0, f"{height} must be multiple of 4"
        self.height = height
        assert width % 4 == 0, f"{width} must be multiple of 4"
        self.width = width

        # computed properties
        # [hp, resources, resources_non_zero, num_planes_player(5),
        # num_planes_unit_type(z), num_planes_unit_action(6)]
        utt = self.interface.utt

        self.float_planes = [1 / MAX_HP, 1 / MAX_RESOURCES, None, None, None, None]
        self.bool_planes = [None, [1], None, None, None, None]
        self.one_hot_planes = [None, None, 3, len(utt["unitTypes"]) + 1, 6, 2]
        if self.interface.partial_obs:
            self.float_planes.append(None)
            self.bool_planes.append(None)
            self.one_hot_planes.append(2)
        self.obs_dim = len(self.float_planes)
        assert self.obs_dim == len(self.bool_planes)
        assert self.obs_dim == len(self.one_hot_planes)
        self.n_dim = (
            len([fp for fp in self.float_planes if fp is not None])
            + sum([len(bp) for bp in self.bool_planes if bp is not None])
            + sum([ohp for ohp in self.one_hot_planes if ohp is not None])
        )

        observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.height, self.width, self.n_dim),
            dtype=np.float32,
        )

        action_space_dims = [6, 4, 4, 4, 4, len(utt["unitTypes"]), 7 * 7]
        action_space = gym.spaces.MultiDiscrete(
            np.array([action_space_dims] * self.height * self.width).flatten().tolist()
        )
        super().__init__(self.interface.num_envs, observation_space, action_space)
        self.action_plane_space = gym.spaces.MultiDiscrete(action_space_dims)

        self.source_unit_idxs = np.tile(
            np.arange(self.height * self.width), (self.num_envs, 1)
        )
        self.source_unit_idxs = self.source_unit_idxs.reshape(
            (self.source_unit_idxs.shape + (1,))
        )
        self.metadata = self.interface.metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self.interface, name)

    def step(self, actions) -> VecEnvStepReturn:
        microrts_action = self._to_microrts_action(actions)
        microrts_obs, microrts_mask, r, d, i = self.interface.step(microrts_action)
        self._update_action_mask(microrts_mask)

        obs = self._from_microrts_obs(microrts_obs)
        return obs, r, d, i

    def reset(self) -> np.ndarray:
        microrts_obs, microrts_mask = self.interface.reset()
        self._update_action_mask(microrts_mask)
        return self._from_microrts_obs(microrts_obs)

    def render(self, mode="human"):
        return self.interface.render(mode)

    def close_extras(self, **kwargs):
        self.interface.close(**kwargs)

    def _translate_actions(
        self, actions: List[List[int]], env_idx: int
    ) -> List[List[int]]:
        map_h = self.interface.heights[env_idx]
        map_w = self.interface.widths[env_idx]
        if map_h == self.height and map_w == self.width:
            return actions
        pad_h = (self.height - map_h) // 2
        pad_w = (self.width - map_w) // 2
        for a in actions:
            y = a[0] // self.width
            x = a[0] % self.width
            a[0] = (y - pad_h) * map_w + x - pad_w
        return actions

    def _verify_actions(self, actions: List[List[int]], env_idx: int):
        matrix_mask = self.interface.debug_matrix_mask(env_idx)
        if matrix_mask is None:
            return
        env_w = self.interface.widths[env_idx]

        if len(actions) != np.sum(matrix_mask[:, :, 0]):
            logging.error(
                f"# Actions mismatch: Env {env_idx}, # Actions {len(actions)} (Should be {np.sum(matrix_mask[:, :, 0])})"
            )
        for a in actions:
            m = matrix_mask[a[0] // env_w, a[0] % env_w]
            if m[0] == 0:
                logging.error(f"No action allowed: Env {env_idx}, loc {a[0]}")
            offset = 1
            for idx, sz in enumerate(self.action_plane_space.nvec):
                valid = m[offset : offset + sz]
                offset += sz
                if np.all(valid == 0):
                    continue
                if not valid[a[idx + 1]]:
                    if idx == 0 or (idx + 1) in ACTION_TYPE_TO_ACTION_INDEXES[a[1]]:
                        logging.error(
                            f"Invalid action in env {env_idx}, loc {a[0]}, action {a[1:]}, idx {idx+1}, valid {valid}"
                        )

    def _to_microrts_action(self, actions: np.ndarray) -> List[List[List[int]]]:
        actions = actions.reshape((self.num_envs, self.height * self.width, -1))
        actions = np.concatenate((self.source_unit_idxs, actions), 2)
        actions = actions[np.where(self.source_unit_mask == 1)]
        action_counts_per_env = self.source_unit_mask.sum(1)

        actions_per_env = []
        action_idx = 0
        for idx, action_count in enumerate(action_counts_per_env):
            actions_in_env = []
            for _ in range(action_count):
                actions_in_env.append(actions[action_idx].tolist())
                action_idx += 1
            actions_per_env.append(self._translate_actions(actions_in_env, idx))
            self._verify_actions(actions_per_env[-1], idx)
        return actions_per_env

    def _from_microrts_obs(self, microrts_obs: List[ByteArray]) -> np.ndarray:
        return np.array(
            [self._encode_obs(o, idx) for idx, o in enumerate(microrts_obs)]
        )

    def _get_matrix_obs(self, obs_bytes: ByteArray, env_idx: int) -> np.ndarray:
        env_h = self.interface.heights[env_idx]
        env_w = self.interface.widths[env_idx]
        obs_array = obs_bytes.reshape((-1, self.obs_dim + 1))

        o = np.zeros((env_h, env_w, self.obs_dim), dtype=obs_array.dtype)
        o[:, :, -1] = self.interface.terrain(env_idx)
        o[obs_array[:, 0], obs_array[:, 1], :-1] = obs_array[:, 2:]

        if env_h == self.height and env_w == self.width:
            return o

        obs = np.zeros((self.height, self.width, self.obs_dim), dtype=o.dtype)
        obs[:, :, -1] = 1
        pad_h = (self.height - env_h) // 2
        pad_w = (self.width - env_w) // 2
        obs[pad_h : pad_h + env_h, pad_w : pad_w + env_w, :] = o
        return obs

    def _verify_matrix_obs(self, obs: np.ndarray, env_idx: int) -> None:
        matrix_obs = self.interface.debug_matrix_obs(env_idx)
        if matrix_obs is None:
            return

        env_h = self.interface.heights[env_idx]
        env_w = self.interface.widths[env_idx]
        pad_h = (self.height - env_h) // 2
        pad_w = (self.width - env_w) // 2
        diffs = np.transpose(
            np.array(
                np.where(
                    matrix_obs != obs[pad_h : pad_h + env_h, pad_w : pad_w + env_w]
                )
            )
        )
        if len(diffs):
            logging.error(f"Observation differences in env {env_idx}: {diffs}")

    def _encode_obs(self, obs_bytes: ByteArray, env_idx: int) -> np.ndarray:
        obs = self._get_matrix_obs(obs_bytes, env_idx)
        self._verify_matrix_obs(obs, env_idx)
        obs = obs.reshape(-1, obs.shape[-1])
        obs_planes = np.zeros((self.height * self.width, self.n_dim), dtype=np.float32)

        plane_idx = 0
        for idx, fp in enumerate(self.float_planes):
            if fp is None:
                continue
            obs_planes[:, plane_idx] = obs[:, idx] * fp
            assert np.all(obs_planes[:, plane_idx] >= 0)
            if np.any(obs_planes[:, plane_idx] > 1):
                warnings.warn(
                    f"Found observations for plane_idx {plane_idx} above max ({np.max(obs[:, idx])})"
                )
            obs_planes[:, plane_idx] = obs_planes[:, plane_idx].clip(0, 1)
            plane_idx += 1
        for idx, bp in enumerate(self.bool_planes):
            if bp is None:
                continue
            for threshold in bp:
                obs_planes[:, plane_idx] = obs[:, idx] >= threshold
                plane_idx += 1
        for idx, ohp in enumerate(self.one_hot_planes):
            if ohp is None:
                continue
            obs_planes[:, plane_idx : plane_idx + ohp] = np.eye(ohp)[
                obs[:, idx].clip(0, ohp)
            ]
            plane_idx += ohp
        assert plane_idx == obs_planes.shape[-1]
        return obs_planes.reshape(self.height, self.width, -1)

    def get_action_mask(self) -> np.ndarray:
        return self._action_mask

    def _update_action_mask(self, microrts_mask: List[ByteArray]) -> None:
        masks = []
        for idx, m_bytes in enumerate(microrts_mask):
            action_plane_dim = np.sum(self.action_plane_space.nvec)
            m_array = m_bytes.reshape(-1, action_plane_dim + 2)
            env_h = self.interface.heights[idx]
            env_w = self.interface.widths[idx]
            m = np.zeros((env_h, env_w, action_plane_dim + 1), dtype=np.bool_)
            m[m_array[:, 0], m_array[:, 1], 0] = 1
            m[m_array[:, 0], m_array[:, 1], 1:] = m_array[:, 2:]
            if env_h == self.height and env_w == self.width:
                masks.append(m)
                continue
            new_m = np.zeros((self.height, self.width, m.shape[-1]), dtype=m.dtype)
            pad_h = (self.height - env_h) // 2
            pad_w = (self.width - env_w) // 2
            new_m[pad_h : pad_h + env_h, pad_w : pad_w + env_w] = m
            masks.append(new_m)
        action_mask = np.array(masks)
        self._verify_action_mask(action_mask)
        self.source_unit_mask = action_mask[:, :, :, 0].reshape(self.num_envs, -1)
        self._action_mask = action_mask[:, :, :, 1:].reshape(
            self.num_envs, self.height * self.width, -1
        )

    def _verify_action_mask(self, masks: np.ndarray) -> None:
        for env_idx, mask in enumerate(masks):
            matrix_mask = self.interface.debug_matrix_mask(env_idx)
            if matrix_mask is None:
                continue

            env_h = self.interface.heights[env_idx]
            env_w = self.interface.widths[env_idx]
            pad_h = (self.height - env_h) // 2
            pad_w = (self.width - env_w) // 2
            diffs = np.transpose(
                np.array(
                    np.where(
                        matrix_mask
                        != mask[pad_h : pad_h + env_h, pad_w : pad_w + env_w]
                    )
                )
            )
            if len(diffs):
                logging.error(f"Mask differences in env {env_idx}: {diffs}")
