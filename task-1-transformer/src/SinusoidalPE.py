import math

import torch
from torch import nn


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if max_len <= 0:
            raise ValueError("max_len must be positive")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        angles = position * frequency

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles[:, : pe[:, 1::2].shape[1]])

        self.d_model = d_model
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("x must have at least 2 dimensions")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"input dimension {x.shape[-1]} does not match d_model={self.d_model}"
            )

        seq_len = x.shape[-2]
        if seq_len > self.pe.shape[0]:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_len={self.pe.shape[0]}"
            )

        pe = self.pe[:seq_len].to(device=x.device, dtype=x.dtype)
        return x + pe
