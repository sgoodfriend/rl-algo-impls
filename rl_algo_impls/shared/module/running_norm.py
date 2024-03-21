import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    def __init__(
        self, num_features: int, clip: float = 10.0, epsilon: float = 1e-8
    ) -> None:
        super().__init__()
        self.clip = clip
        self.epsilon = epsilon

        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("mean", torch.zeros(num_features, dtype=torch.float64))
        self.register_buffer("var", torch.ones(num_features, dtype=torch.float64))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.training:
                self.update(x, mask)

            normalized = torch.clamp(
                (x - self.mean.float()) / torch.sqrt(self.var.float() + self.epsilon),
                -self.clip,
                self.clip,
            )
        return normalized

    def update(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        masked_x = x[mask]
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
