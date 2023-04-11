from typing import Callable

from rl_algo_impls.shared.callbacks import Callback
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper


class SelfPlayCallback(Callback):
    def __init__(
        self,
        policy: Policy,
        policy_factory: Callable[[], Policy],
        selfPlayWrapper: SelfPlayWrapper,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.policy_factory = policy_factory
        self.selfPlayWrapper = selfPlayWrapper
        self.checkpoint_policy()

    def on_step(self, timesteps_elapsed: int = 1) -> bool:
        super().on_step(timesteps_elapsed)
        if (
            self.timesteps_elapsed
            >= self.last_checkpoint_step + self.selfPlayWrapper.save_steps
        ):
            self.checkpoint_policy()
        return True

    def checkpoint_policy(self):
        self.selfPlayWrapper.checkpoint_policy(
            self.policy_factory().load_from(self.policy)
        )
        self.last_checkpoint_step = self.timesteps_elapsed
