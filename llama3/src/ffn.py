import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()

        hidden_dim = int(hidden_dim * 2 / 3)

        self.linear1 = nn.Linear(dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(dim, hidden_dim, bias=False)
        self.linear3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear3(nn.functional.silu(self.linear1(x)) * self.linear2(x))
