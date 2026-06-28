"""TinyTCN — lightweight causal TCN for 12-channel bilateral IMU gait phase classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(nn.functional.pad(x, (self.pad, 0)))
        out = self.act(self.bn(out))
        return out + self.residual(x)


class TinyTCN(nn.Module):
    """Input (B, T, C) → logits (B, 4) for LR / LS / PSw / Sw."""

    def __init__(self, n_channels: int = 12, n_classes: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            CausalConvBlock(n_channels, hidden, kernel=5, dilation=1),
            CausalConvBlock(hidden, hidden, kernel=5, dilation=2),
            CausalConvBlock(hidden, hidden, kernel=3, dilation=4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        return self.head(self.stem(x))


def build_model(n_channels: int, n_classes: int = 4, hidden: int = 32, **_) -> TinyTCN:
    """Training entry point — shared `build_model` convention."""
    return TinyTCN(n_channels=n_channels, n_classes=n_classes, hidden=hidden)
