from typing import Optional

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    def __init__(
        self,
        num_features: Optional[int],
        feature_indexes: Optional[torch.Tensor] = None,
        clip: float = 10.0,
        epsilon: float = 1e-8,
    ) -> None:
        assert bool(num_features) != bool(
            feature_indexes is not None
        ), "One (and only one) of num_features and feature_indexes should be provided"
        super().__init__()
        self.feature_indexes = feature_indexes
        self.clip = clip
        self.epsilon = epsilon

        num_features = (
            feature_indexes.size(0) if feature_indexes is not None else num_features
        )
        assert isinstance(num_features, int)

        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("mean", torch.zeros(num_features, dtype=torch.float64))
        self.register_buffer("var", torch.ones(num_features, dtype=torch.float64))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.training:
                self.update(x, mask)

            if self.feature_indexes is not None:
                x[..., self.feature_indexes] = torch.clamp(
                    (x[..., self.feature_indexes] - self.mean.float())
                    / torch.sqrt(self.var.float() + self.epsilon),
                    -self.clip,
                    self.clip,
                )
            else:
                x = torch.clamp(
                    (x - self.mean.float())
                    / torch.sqrt(self.var.float() + self.epsilon),
                    -self.clip,
                    self.clip,
                )
        return x

    def update(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        masked_x = x[mask]
        if self.feature_indexes is not None:
            masked_x = masked_x[:, self.feature_indexes]
        batch_mean = masked_x.mean(dim=0)
        batch_var = masked_x.var(dim=0)
        batch_count = masked_x.size(0)

        delta = batch_mean.double() - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var.double() * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
