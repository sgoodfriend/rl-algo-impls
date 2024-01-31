from abc import ABC, abstractmethod
from typing import Optional, Tuple

from rl_algo_impls.shared.policy.policy import Policy


class CheckpointPoliciesDelegate(ABC):
    @abstractmethod
    def update_checkpoint_policies(
        self,
        checkpoint_policies: Tuple[Policy, ...],
        latest_checkpoint_idx: int,
    ) -> None:
        ...
