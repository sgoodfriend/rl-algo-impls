from copy import deepcopy
from typing import Callable

from rl_algo_impls.shared.callbacks import Callback
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper


class SelfPlayCallback(Callback):
    def __init__(
        self,
        policy: Policy,
        self_play_wrapper: SelfPlayWrapper,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.self_play_wrapper = self_play_wrapper
        self.checkpoint_policy()

    def on_step(self, timesteps_elapsed: int = 1, **kwargs) -> bool:
        super().on_step(timesteps_elapsed)
        if (
            self.timesteps_elapsed
            >= self.last_checkpoint_step + self.self_play_wrapper.save_steps
        ):
            self.checkpoint_policy()
        return True

    def checkpoint_policy(self):
        self.self_play_wrapper.checkpoint_policy(deepcopy(self.policy))
        self.last_checkpoint_step = self.timesteps_elapsed
