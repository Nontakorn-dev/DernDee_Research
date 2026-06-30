"""Hybrid LSTM+GRU baseline for gait phase classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMGRU(nn.Module):
    """Input (B, T, C) -> logits (B, n_classes)."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int = 4,
        hidden: int = 32,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(lstm_out)
        return self.head(gru_out[:, -1, :])


def build_model(
    n_channels: int,
    n_classes: int = 4,
    hidden: int = 32,
    num_layers: int = 1,
    **_,
) -> LSTMGRU:
    return LSTMGRU(
        n_channels=n_channels,
        n_classes=n_classes,
        hidden=hidden,
        num_layers=num_layers,
    )
