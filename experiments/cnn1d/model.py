"""1D-CNN baseline for gait phase classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """Input (B, T, C) -> logits (B, n_classes)."""

    def __init__(self, n_channels: int, n_classes: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        return self.head(self.features(x))


def build_model(n_channels: int, n_classes: int = 4, hidden: int = 32, **_) -> CNN1D:
    return CNN1D(n_channels=n_channels, n_classes=n_classes, hidden=hidden)
