"""Decoder-only mini-GPT model interfaces."""

# %%
from __future__ import annotations
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from torch import Tensor, nn

from src.rope import RoPE
from src.attention import KVCache, CausalSelfAttention
from src.tokenizer import BPETokenizer


ModelKVCache: TypeAlias = list[KVCache]


# import torch
# a = torch.Tensor([2,4,4])
# probs = torch.tensor([0.2, 0.3, 0.5])
# idx = torch.multinomial(probs, num_samples=1)       # scalar
# idx = torch.multinomial(probs, num_samples=5, replacement=False)  # [5]
# idx

@dataclass
class MiniGPTConfig:
    """Hyperparameters required to construct a MiniGPT model."""

    vocab_size: int
    block_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    embed_dim: int = 128
    dropout: float = 0.2
    ffn_multiplier: int = 2
    rope_base: float = 10_000.0


class FeedForward(nn.Module):
    """Position-wise feed-forward network used inside each decoder block."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.W1 = nn.Linear(embed_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, embed_dim)

        self.GELU = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """Transform an input tensor while preserving its ``[B, T, D]`` shape."""

        x = self.W1(x)
        x = self.GELU(x)
        x = self.W2(x)
        x = self.dropout(x)

        return x


class DecoderBlock(nn.Module):
    """Pre-LN causal Transformer decoder block."""

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(
            config.embed_dim,
            config.num_heads,
            config.block_size,
            config.dropout,
            config.rope_base,
        )
        self.ln1 = nn.LayerNorm(config.embed_dim)
        # self.register_buffer("kv_chache", KVCache())
        self.ffn = FeedForward(
            config.embed_dim, config.ffn_multiplier * config.embed_dim, config.dropout
        )
        self.ln2 = nn.LayerNorm(config.embed_dim)

        # self.dropout1 = nn.Dropout(config.dropout)
        # self.dropout2 = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        kv_cache: KVCache | None = None,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KVCache]:
        """Apply attention and feed-forward residual branches."""

        if return_cache:
            attn, new_cache = self.attn(self.ln1(x), kv_cache, return_cache)
        else:
            attn = self.attn(self.ln1(x), kv_cache, return_cache)

        x = x + (attn)

        x = x + (self.ffn(self.ln2(x)))

        if return_cache:
            return x, new_cache
        else:
            return x


class MiniGPT(nn.Module):
    """Decoder-only language model that predicts the next token."""

    block_size: int

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()

        self.block_size = config.block_size

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_layers)]
        )

        self.ln = nn.LayerNorm(config.embed_dim)

        self.classify = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(
        self,
        ids: Tensor,
        kv_cache: ModelKVCache | None = None,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, ModelKVCache]:
        """Return next-token logits and optionally an updated per-layer cache.

        ``ids`` has shape ``[B, T]`` and logits have shape ``[B, T, vocab_size]``.
        """

        if kv_cache is not None and len(kv_cache) != len(self.blocks):
            raise ValueError("incorrect cache numbers")

        x = self.embedding(ids)

        modelkv_caches = []

        for index, block in enumerate(self.blocks):
            decoder: DecoderBlock = block

            last_kv_cache = None if kv_cache is None else kv_cache[index]
            if return_cache:
                x, last_kv_cache = decoder(x, last_kv_cache, return_cache)
            else:
                x = decoder(x, last_kv_cache, return_cache)
            modelkv_caches.append(last_kv_cache)

        x = self.ln(x)

        x = self.classify(x)

        if return_cache:
            return x, modelkv_caches
        else:
            return x

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
    ) -> Tensor:
        """Autoregressively extend prompt IDs using the requested sampling strategy."""
        self.eval()
        if max_new_tokens + prompt_ids.shape[1] > self.block_size:
            raise ValueError("Too many tokens")

        loop_ids = prompt_ids
        if temperature == 0:
            kv_cache = None
            for i in range(max_new_tokens):
                token, kv_cache = self(loop_ids, kv_cache, True)
                last_token = token[:, -1, :]
                last_token, token_indice = torch.topk(last_token, 1, dim=-1)
                prompt_ids = torch.cat([prompt_ids, token_indice], dim=-1)
                loop_ids = token_indice

        else:
            kv_cache = None
            for i in range(max_new_tokens):
                token, kv_cache = self(loop_ids, kv_cache, True)
                last_token = token[:, -1, :]
                last_token /= temperature
                last_token = torch.softmax(last_token, dim=-1)
                # last_token, token_indice = torch.topk(last_token, top_k, dim=-1)
                # prompt_ids = torch.cat([prompt_ids, token_indice], dim=-1)

                sorted_vals, sorted_indices = torch.sort(
                    last_token, dim=-1, descending=True
                )

                if top_k is not None:
                    sorted_indices = sorted_indices[:, :top_k]
                    sorted_vals = sorted_vals[:, :top_k]
                    sorted_vals = sorted_vals / sorted_vals.sum(-1, True)

                if top_p is not None:
                    cumsum = torch.cumsum(sorted_vals, dim=-1)
                    pos = (cumsum >= top_p).bool()
                    pos[:, 1:] = pos[:, :-1].clone()
                    pos[:, 0] = False
                    sorted_vals = sorted_vals.masked_fill(pos, 0)

                idx = torch.multinomial(sorted_vals, 1)

                token_to_return = sorted_indices.gather(1, idx)
                prompt_ids = torch.cat([prompt_ids, token_to_return], dim=-1)
                loop_ids = token_to_return
        return prompt_ids


def load_for_eval(ckpt_path: str | Path) -> tuple[MiniGPT, BPETokenizer]:
    """Restore a trained model and its matching tokenizer for official evaluation."""
    path = Path(ckpt_path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    token_path = parent / "tokenizer.json"
    bpe = BPETokenizer().from_pretrained(token_path)
    model_config_path = parent / "model_config.json"

    import json
    model_config = json.loads(model_config_path.read_text("utf-8"))
    model = MiniGPT(MiniGPTConfig(**model_config))

    state = torch.load(path, "cpu")
    model.load_state_dict(state)
    model.eval()

    return model, bpe
# %%
