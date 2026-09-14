import math

import torch
from torch import nn
from torch.nn import functional as F

from src.rope import RoPe


class GroupedQueryAttention(nn.Module):
    def __init__(self, input_size: int, head_dim: int, output_size: int,
                 rope_encoder: RoPe, num_heads: int = 8, num_kv_heads: int = 4,
                 is_masked: bool = False) -> None:
        super().__init__()

        self.rope_encoder = rope_encoder

        self.input_size = input_size
        self.head_dim = head_dim
        self.output_size = output_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.is_masked = is_masked

        self.query = nn.Linear(self.input_size, self.head_dim * self.num_heads, bias=False)
        self.key = nn.Linear(self.input_size, self.head_dim * self.num_kv_heads, bias=False)
        self.value = nn.Linear(self.input_size, self.head_dim * self.num_kv_heads, bias=False)
        self.output = nn.Linear(self.head_dim *self. num_heads, self.output_size, bias=False)

    @staticmethod
    def repeat_kv(tensor: torch.Tensor, repeat: int) -> torch.Tensor:
        s = tensor.shape
        return tensor[:, :, None].expand(s[0], s[1], repeat, s[2], s[3]).reshape(s[0], s[1] * repeat, s[2], s[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        length = x.size(1)
        q = self.query(x).view(batch, length, self.num_heads, self.head_dim)
        k = self.key(x).view(batch, length, self.num_kv_heads, self.head_dim)
        v = self.value(x).view(batch, length, self.num_kv_heads, self.head_dim)

        rotated = self.rope_encoder.rotate(q, k, 0)
        q, k = rotated.queries, rotated.keys

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        repeat = self.num_heads // self.num_kv_heads
        k = GroupedQueryAttention.repeat_kv(k, repeat)
        v = GroupedQueryAttention.repeat_kv(v, repeat)

        attention_weights = q @ k.transpose(2, 3) / math.sqrt(self.head_dim)
        if self.is_masked:
            mask = torch.tril(torch.ones(length, length, device=self.device).expand(batch, 1, length, length)).bool()
            attention_weights = attention_weights.masked_fill(~mask, -torch.inf)

        attention_weights = F.softmax(attention_weights.float(), dim=-1).type_as(q)
        result = ((attention_weights @ v)
                  .transpose(1, 2)
                  .reshape(batch, length, self.num_heads * self.head_dim))

        return self.output(result)
