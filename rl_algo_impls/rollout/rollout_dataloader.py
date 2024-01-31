from typing import List, Union

import torch
from torch.utils.data import DataLoader, Dataset

from rl_algo_impls.shared.tensor_utils import TDN, BatchTuple


class RolloutDataset(Dataset):
    batch: BatchTuple

    def __init__(self, batch: BatchTuple) -> None:
        self.batch = batch

    def __getitem__(self, indices: Union[int, List[int], torch.Tensor]) -> BatchTuple:
        def by_indices_fn(_t: TDN) -> TDN:
            if _t is None:
                return _t
            if isinstance(_t, dict):
                return {k: v[indices] for k, v in _t.items()}
            return _t[indices]

        return self.batch.__class__(*(by_indices_fn(t) for t in self.batch))

    def __getitems__(self, indices: Union[int, List[int], torch.Tensor]) -> BatchTuple:
        return self.__getitem__(indices)

    def __len__(self):
        return self.batch.obs.shape[0]  # type: ignore


class RolloutDataLoader(DataLoader):
    def __init__(self, dataset: RolloutDataset, **kwargs) -> None:
        super().__init__(
            dataset,
            collate_fn=lambda x: x,  # Handled by RolloutDataset.__getitems__
            **kwargs,
        )
