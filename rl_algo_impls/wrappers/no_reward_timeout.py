from typing import Optional, SupportsFloat, Tuple, Union

import gymnasium
import numpy as np

ObsType = Union[np.ndarray, dict]
ActType = Union[int, float, np.ndarray, dict]


class NoRewardTimeout(gymnasium.Wrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        n_timeout_steps: int,
        n_fire_steps: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        self.n_timeout_steps = n_timeout_steps
        self.n_fire_steps = n_fire_steps

        self.fire_act = None
        if n_fire_steps is not None:
            action_meanings = env.unwrapped.get_action_meanings()
            assert "FIRE" in action_meanings
            self.fire_act = action_meanings.index("FIRE")

        self.steps_since_reward = 0

        self.episode_score = 0.0
        self.episode_step_idx = 0

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, dict]:
        if self.steps_since_reward == self.n_fire_steps:
            assert self.fire_act is not None
            self.print_intervention("Force fire action")
            action = self.fire_act
        obs, rew, terminated, truncated, info = self.env.step(action)

        self.episode_score += float(rew)
        self.episode_step_idx += 1

        if rew != 0 or terminated or truncated:
            self.steps_since_reward = 0
        else:
            self.steps_since_reward += 1
            if self.steps_since_reward >= self.n_timeout_steps:
                self.print_intervention("Early terminate")
                truncated = True

        return obs, rew, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        self._reset_state()
        return self.env.reset(**kwargs)

    def _reset_state(self) -> None:
        self.steps_since_reward = 0
        self.episode_score = 0
        self.episode_step_idx = 0

    def print_intervention(self, tag: str) -> None:
        logging.info(
            f"{self.__class__.__name__}: {tag} | "
            f"Score: {self.episode_score} | "
            f"Length: {self.episode_step_idx}"
        )
