import math

import torch
from torch import nn
from torch.nn import functional as F

from src.rope import RoPe


class GroupedQueryAttention(nn.Module):
    def __init__(self, input_size: int, head_dim: int, output_size: int,
                 rope_encoder: RoPe, max_batch_size: int, max_seq_length: int,
                 num_heads: int = 8, num_kv_heads: int = 4, is_masked: bool = False) -> None:
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

        shape = (max_batch_size, max_seq_length, num_kv_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(shape), persistent=False)

    @staticmethod
    def repeat_kv(tensor: torch.Tensor, repeat: int) -> torch.Tensor:
        s = tensor.shape
        return tensor[:, :, None].expand(s[0], s[1], repeat, s[2], s[3]).reshape(s[0], s[1] * repeat, s[2], s[3])

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        batch = x.size(0)
        length = x.size(1)
        q = self.query(x).view(batch, length, self.num_heads, self.head_dim)
        k = self.key(x).view(batch, length, self.num_kv_heads, self.head_dim)
        v = self.value(x).view(batch, length, self.num_kv_heads, self.head_dim)

        rotated = self.rope_encoder.rotate(q, k, start_pos)
        q, k = rotated.queries, rotated.keys

        if not self.training:
            self.k_cache[ :batch, start_pos : start_pos + length] = k
            self.v_cache[ :batch, start_pos : start_pos + length] = v
            k = self.k_cache[:batch, : start_pos + length]
            v = self.v_cache[:batch, : start_pos + length]


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
