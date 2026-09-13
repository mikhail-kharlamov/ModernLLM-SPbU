import torch


class RoPe:
    def __init__(self, max_tokens_length: int, embedding_length: int,
                 head_dim: int, theta: float = 500000.0) -> None:
        self.theta = theta
        self.embedding_length = embedding_length

        freqs = theta ** (-1 * (torch.arange(0, head_dim, 2) / head_dim))
        arange = torch.arange(max_tokens_length, dtype=torch.float32)
        angels = torch.outer(arange, freqs)
        self.rotation = torch.polar(torch.ones_like(angels), angels)

    def number(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        query_pairs = torch.view_as_complex(query)
        key_pairs = torch.view_as_complex(key)
