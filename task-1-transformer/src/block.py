from torch import nn

from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim=None, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, n_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.layer_norm2 = nn.LayerNorm(d_model)

        d_inside = 4 * d_model if ff_dim is None else ff_dim
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_inside),
            nn.GELU(),
            nn.Linear(d_inside, d_model)
        )

        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x, mask=None):

        #first layer
        y = x + self.dropout1(self.attn(self.layer_norm1(x), mask))

        #second layer
        z = y + self.dropout2(self.ffn(self.layer_norm2(y)))

        return z
