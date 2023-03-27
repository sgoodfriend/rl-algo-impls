from typing import TypeVar

import numpy as np
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from jpype.types import JArray, JInt

from rl_algo_impls.wrappers.vectorable_wrapper import VecEnvStepReturn

MicroRTSGridModeVecEnvCompatSelf = TypeVar(
    "MicroRTSGridModeVecEnvCompatSelf", bound="MicroRTSGridModeVecEnvCompat"
)


class MicroRTSGridModeVecEnvCompat(MicroRTSGridModeVecEnv):
    def step(self, action: np.ndarray) -> VecEnvStepReturn:
        indexed_actions = np.concatenate(
            [
                np.expand_dims(
                    np.stack(
                        [np.arange(0, action.shape[1]) for i in range(self.num_envs)]
                    ),
                    axis=2,
                ),
                action,
            ],
            axis=2,
        )
        action_mask = np.array(self.vec_client.getMasks(0), dtype=np.bool8).reshape(
            indexed_actions.shape[:-1] + (-1,)
        )
        valid_action_mask = action_mask[:, :, 0]
        valid_actions_counts = valid_action_mask.sum(1)
        valid_actions = indexed_actions[valid_action_mask]
        valid_actions_idx = 0

        all_valid_actions = []
        for env_act_cnt in valid_actions_counts:
            env_valid_actions = []
            for _ in range(env_act_cnt):
                env_valid_actions.append(JArray(JInt)(valid_actions[valid_actions_idx]))
                valid_actions_idx += 1
            all_valid_actions.append(JArray(JArray(JInt))(env_valid_actions))
        return super().step(JArray(JArray(JArray(JInt)))(all_valid_actions))  # type: ignore

    @property
    def unwrapped(
        self: MicroRTSGridModeVecEnvCompatSelf,
    ) -> MicroRTSGridModeVecEnvCompatSelf:
        return self
