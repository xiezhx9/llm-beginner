"""Causal multi-head self-attention interfaces for mini-GPT."""

# %%
from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor, nn

from .rope import RoPE

KVCache: TypeAlias = tuple[Tensor, Tensor]


# %%
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and KV-cache support."""

    embed_dim: int
    num_heads: int
    head_dim: int
    max_seq_len: int

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        rope_base: float = 10_000.0,
    ) -> None:
        """Configure QKV projections, RoPE, attention dropout, and output projection."""
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.rope_base = rope_base

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim cant be divide by num_heads")
        self.head_dim = embed_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("Head Dim must be even")

        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.Wo = nn.Linear(embed_dim, embed_dim)

        self.rope = RoPE(self.head_dim, self.rope_base, 2048)

        self.dropout_module = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        """Convert ``[B, T, D]`` into ``[B, H, T, head_dim]``."""
        B, T, D = x.shape
        x = x.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        return x

    def _merge_heads(self, x: Tensor) -> Tensor:
        """Convert ``[B, H, T, head_dim]`` back into ``[B, T, D]``."""
        B, H, T, Head_dim = x.shape

        x = x.transpose(1, 2).reshape(B, T, H * Head_dim)
        return x

    @staticmethod
    def _build_causal_mask(
        query_len: int,
        key_len: int,
        past_len: int,
        device: torch.device,
    ) -> Tensor:
        """Build a broadcastable mask of shape ``[1, 1, Tq, Tk]``."""
        col_tensor = torch.arange(query_len).to(device).unsqueeze(-1) + past_len
        row_tensor = torch.arange(key_len).to(device).unsqueeze(0)

        return (col_tensor >= row_tensor).bool()[None, None, :, :]

    def forward(
        self,
        x: Tensor,
        kv_cache: KVCache | None = None,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KVCache]:
        """Run causal self-attention and optionally return the updated K/V cache.

        ``x`` has shape ``[B, T, D]``. Cached keys and values use
        ``[B, H, T_past, head_dim]`` and grow along the sequence dimension.
        """

        T_past = 0 if kv_cache is None else kv_cache[0].shape[2]
        all_T = x.shape[1] if kv_cache is None else x.shape[1] + kv_cache[0].shape[2]

        casual_mask = CausalSelfAttention._build_causal_mask(
            x.shape[1], all_T, T_past, x.device
        )

        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)

        Qk = self._split_heads(Q)
        Kk = self._split_heads(K)
        Vk = self._split_heads(V)

        Qk_rope, Kk_rope = self.rope(Qk, Kk, T_past)

        Kk_all = (
            Kk_rope if kv_cache is None else torch.cat([kv_cache[0], Kk_rope], dim=-2)
        )
        Vk_all = Vk if kv_cache is None else torch.cat([kv_cache[1], Vk], dim=-2)

        attn = Qk_rope @ Kk_all.transpose(-1, -2) / self.head_dim**0.5
        attn = torch.softmax(attn.masked_fill(~casual_mask, float("-inf")), dim=-1)
        attn = self.dropout_module(attn)
        attn = attn @ Vk_all

        attn = attn.transpose(1, 2).reshape(x.shape)

        attn = self.Wo(attn)

        if return_cache:
            return attn, (Kk_all, Vk_all)
        else:
            return attn
