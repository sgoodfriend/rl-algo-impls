from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rl_algo_impls.shared.data_store.data_store_data import CheckpointState
from rl_algo_impls.shared.trackable import Trackable

if TYPE_CHECKING:
    from rl_algo_impls.shared.algorithm import Algorithm
    from rl_algo_impls.shared.policy.policy import Policy


class InProcessDataStore:
    def __init__(self, checkpoint_history_size: int):
        self._policy: Optional["Policy"] = None
        self.env_trackers: Dict[str, Trackable] = {}
        self.timesteps_elapsed = 0
        self.rollout_params: Dict[str, Any] = {}
        self.load_path: Optional[str] = None

        self.checkpoint_history_size = checkpoint_history_size
        self._ckpts_circular_queue: List[CheckpointState] = []
        self._latest_ckpt_idx = -1

    @property
    def checkpoints(self) -> List[CheckpointState]:
        return list(self._ckpts_circular_queue)

    @property
    def latest_checkpoint(self) -> Optional[CheckpointState]:
        return (
            self._ckpts_circular_queue[self._latest_ckpt_idx]
            if self._ckpts_circular_queue
            else None
        )

    @property
    def policy(self) -> "Policy":
        assert self._policy is not None
        return self._policy

    @policy.setter
    def policy(self, policy: "Policy") -> None:
        if self._policy is None and self.load_path is not None:
            policy.load(self.load_path)
        self._policy = policy

    def create_checkpoint(self, checkpoint_state: CheckpointState) -> None:
        if self.checkpoint_history_size == 0:
            return
        if len(self._ckpts_circular_queue) < self.checkpoint_history_size:
            self._latest_ckpt_idx = len(self._ckpts_circular_queue)
            self._ckpts_circular_queue.append(checkpoint_state)
        else:
            self._latest_ckpt_idx = (
                self._latest_ckpt_idx + 1
            ) % self.checkpoint_history_size
            self._ckpts_circular_queue[self._latest_ckpt_idx] = checkpoint_state

    def load(self, path: str) -> None:
        self.load_path = path
        if self._policy is not None:
            self._policy.load(path)
        for tracker in self.env_trackers.values():
            tracker.get_state().load(path)
