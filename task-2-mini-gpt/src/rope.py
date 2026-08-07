"""Rotary position embedding interface for the mini-GPT model."""

# %%
from __future__ import annotations

import torch
from torch import Tensor, nn

# %%


class RoPE(nn.Module):
    """Apply rotary position embeddings to attention queries and keys.

    Query and key tensors use the shape ``[B, H, T, head_dim]``. Adjacent
    feature dimensions form each two-dimensional rotation pair.
    """

    head_dim: int
    base: float
    max_seq_len: int

    def __init__(
        self,
        head_dim: int,
        base: float = 10_000.0,
        max_seq_len: int = 2_048,
    ) -> None:
        """Initialize RoPE configuration and its reusable frequency cache."""
        super().__init__()

        if head_dim % 2 != 0:
            raise AttributeError("head dim must be even")

        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len

        row_tensor = torch.arange(max_seq_len).unsqueeze(1).to(torch.float32)

        col_tensor = self.base ** (
            -torch.arange(0, head_dim, step=2).unsqueeze(0) / head_dim
        )

        arg_tensor = row_tensor @ col_tensor

        self.register_buffer("cos_arg_tensor", arg_tensor.cos(), persistent=False)
        self.register_buffer("sin_arg_tensor", arg_tensor.sin(), persistent=False)

    def _get_cos_sin(
        self,
        seq_len: int,
        position_offset: int,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Return broadcastable cosine and sine values for requested positions.

        Both returned tensors have shape ``[1, 1, T, head_dim // 2]``.
        """

        if position_offset + seq_len > self.max_seq_len:
            raise AttributeError("KV offset beyond the max seq len")

        result_tensor1 = self.cos_arg_tensor.to(dtype)[
            None, None, position_offset : position_offset + seq_len, :
        ]
        result_tensor2 = self.sin_arg_tensor.to(dtype)[
            None, None, position_offset : position_offset + seq_len, :
        ]

        return result_tensor1, result_tensor2

    @staticmethod
    def _apply_rotation(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Rotate adjacent feature pairs in one ``[B, H, T, head_dim]`` tensor."""

        even_cos_tensor = x[..., ::2] * cos
        even_sin_tensor = x[..., ::2] * sin
        odd_cos_tensor = x[..., 1::2] * cos
        odd_sin_tensor = x[..., 1::2] * sin

        result = torch.empty_like(x)

        result[..., ::2] = even_cos_tensor - odd_sin_tensor
        result[..., 1::2] = even_sin_tensor + odd_cos_tensor

        return result

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        position_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Apply the same position-dependent rotations to Q and K.

        ``position_offset`` is the number of tokens already stored in the KV
        cache and determines the first position assigned to the current input.
        """

        cos, sin = self._get_cos_sin(q.shape[2], position_offset, q.dtype)

        qk = RoPE._apply_rotation(q, cos, sin)
        kk = RoPE._apply_rotation(k, cos, sin)

        return qk, kk
