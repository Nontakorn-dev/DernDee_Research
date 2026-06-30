"""Small Transformer encoder baseline for gait phase classification."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerGait(nn.Module):
    """Input (B, T, C) -> logits (B, n_classes)."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int = 4,
        hidden: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError(f"hidden={hidden} must be divisible by num_heads={num_heads}")

        self.input_proj = nn.Linear(n_channels, hidden)
        self.pos_encoder = PositionalEncoding(hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :])


def build_model(
    n_channels: int,
    n_classes: int = 4,
    hidden: int = 32,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    **_,
) -> TransformerGait:
    return TransformerGait(
        n_channels=n_channels,
        n_classes=n_classes,
        hidden=hidden,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )
