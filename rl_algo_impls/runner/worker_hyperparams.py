from dataclasses import dataclass


@dataclass
class WorkerHyperparams:
    n_rollout_workers: int = 1
    rollout_gpu_index: int = 0
    evaluator_gpu_index: int = 0

    @property
    def desired_num_accelerators(self) -> int:
        return max(self.rollout_gpu_index, self.evaluator_gpu_index) + 1
