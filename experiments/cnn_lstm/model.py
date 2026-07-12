"""CNN-LSTM hybrid baseline for gait phase classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """Input (B, T, C) -> logits (B, n_classes)."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int = 4,
        hidden: int = 32,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        return self.head(lstm_out[:, -1, :])


def build_model(
    n_channels: int,
    n_classes: int = 4,
    hidden: int = 32,
    num_layers: int = 1,
    **_,
) -> CNNLSTM:
    return CNNLSTM(
        n_channels=n_channels,
        n_classes=n_classes,
        hidden=hidden,
        num_layers=num_layers,
    )
