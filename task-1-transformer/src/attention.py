import math

import torch
from torch import nn


def _attention_weights(Q, K, mask=None):
    assert Q.shape[-1] == K.shape[-1]
    dk = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / math.sqrt(dk)

    if mask is not None:
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    return torch.softmax(scores, dim=-1)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute scaled dot-product attention.

    Q, K, V: (batch, heads, seq_len, head_dim)
    mask: broadcastable to (batch, heads, seq_len, seq_len), True means masked.

    target : nn.functional.scaled_dot_product_attention
    Attention(Q, K, V) = softmax( (Q · K的转置) / √dk ) · V
    """

    return _attention_weights(Q, K, mask) @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads=4):

        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0
        self.dk = self.d_model // n_heads

        self.Wq = nn.Linear(self.d_model, self.d_model)
        self.Wk = nn.Linear(self.d_model, self.d_model)
        self.Wv = nn.Linear(self.d_model, self.d_model)
        self.Wo = nn.Linear(self.d_model, self.d_model)

    def forward(self, x: torch.Tensor, mask=None, return_weights=False):

        Q = self.Wq(x)
        Q_mulhead = Q.reshape(Q.shape[0], Q.shape[1], self.n_heads, self.dk)
        Q_mulhead = Q_mulhead.transpose(1, 2)


        K = self.Wk(x)
        K_mulhead = K.reshape(K.shape[0], K.shape[1], self.n_heads, self.dk)
        K_mulhead = K_mulhead.transpose(1, 2)

        V = self.Wv(x)
        V_mulhead = V.reshape(V.shape[0], V.shape[1], self.n_heads, self.dk)
        V_mulhead = V_mulhead.transpose(1, 2)

        weights = _attention_weights(Q_mulhead, K_mulhead, mask)
        attn = weights @ V_mulhead
        attn = attn.transpose(1, 2).reshape(x.shape)
        output = self.Wo(attn)

        if return_weights:
            return output, weights
        return output
