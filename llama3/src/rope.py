from dataclasses import dataclass

import torch


@dataclass
class RotatedVectors:
    keys: torch.Tensor
    queries: torch.Tensor


class RoPe:
    def __init__(self, max_tokens_length: int, head_dim: int, theta: float = 500000.0) -> None:
        self.theta = theta

        freqs = theta ** (-1 * (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        arange = torch.arange(max_tokens_length, dtype=torch.float32)
        angles = torch.outer(arange, freqs)
        self.rotation = torch.polar(torch.ones_like(angles), angles)

    def number(self, query: torch.Tensor, key: torch.Tensor, start_token_pos: int) -> RotatedVectors:
        length = query.size(1)
        rotation_matrix = self.rotation[start_token_pos : start_token_pos + length][None, :, None, :]
        query_pairs = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2)) * rotation_matrix
        key_pairs = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2)) * rotation_matrix

        return RotatedVectors(keys=torch.view_as_real(key_pairs).flatten(3),
                              queries=torch.view_as_real(query_pairs).flatten(3))
