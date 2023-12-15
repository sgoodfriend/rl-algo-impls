from copy import deepcopy
from typing import List, Optional, Sequence

from rl_algo_impls.shared.policy.policy import Policy


class PolicyCheckpointsManager:
    def __init__(self, history_size: int) -> None:
        self.history_size = history_size
        self._ckpts_circular_queue: List[Policy] = []
        self._latest_ckpt_idx = -1

    @property
    def checkpoints(self) -> Sequence[Policy]:
        return list(self._ckpts_circular_queue)

    @property
    def latest_checkpoint(self) -> Optional[Policy]:
        return (
            self._ckpts_circular_queue[self._latest_ckpt_idx]
            if self._ckpts_circular_queue
            else None
        )

    def create_checkpoint(self, policy: Policy) -> None:
        checkpoint = deepcopy(policy)
        checkpoint.eval()
        if len(self._ckpts_circular_queue) < self.history_size:
            self._latest_ckpt_idx = len(self._ckpts_circular_queue)
            self._ckpts_circular_queue.append(checkpoint)
        else:
            self._latest_ckpt_idx = (self._latest_ckpt_idx + 1) % self.history_size
            self._ckpts_circular_queue[self._latest_ckpt_idx] = checkpoint
