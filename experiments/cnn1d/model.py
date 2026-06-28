"""1D-CNN baseline for gait phase classification — TODO."""

from __future__ import annotations

import torch.nn as nn


def build_model(n_channels: int, n_classes: int = 4, hidden: int = 32, **_) -> nn.Module:
    raise NotImplementedError(
        "Implement CNN1D in experiments/cnn1d/model.py. "
        "Must accept (B, T, C) input and return (B, n_classes) logits."
    )
