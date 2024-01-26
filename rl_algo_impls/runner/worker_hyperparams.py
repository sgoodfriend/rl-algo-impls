from dataclasses import dataclass
from typing import List, Optional


@dataclass
class WorkerHyperparams:
    n_rollout_workers: int = 1
    rollout_gpu_indexes: Optional[List[int]] = None
    evaluator_gpu_index: int = 0
    n_inference_workers: int = 1
    inference_gpu_indexes: Optional[List[int]] = None
    different_seeds_for_rollout_workers: bool = True

    @property
    def desired_num_accelerators(self) -> int:
        max_rollout_gpu_index = (
            max(self.rollout_gpu_indexes) if self.rollout_gpu_indexes else 0
        )
        max_inference_gpu_index = (
            max(self.inference_gpu_indexes) if self.inference_gpu_indexes else 0
        )
        return (
            max(
                max_rollout_gpu_index,
                self.evaluator_gpu_index,
                max_inference_gpu_index,
            )
            + 1
        )

    def rollout_gpu_index(self, rollout_worker_idx: int) -> int:
        if self.rollout_gpu_indexes:
            return self.rollout_gpu_indexes[
                rollout_worker_idx % len(self.rollout_gpu_indexes)
            ]
        return 0
