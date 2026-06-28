"""Transformer encoder baseline — TODO."""

from __future__ import annotations

import torch.nn as nn


def build_model(n_channels: int, n_classes: int = 4, hidden: int = 32, **_) -> nn.Module:
    raise NotImplementedError("Implement Transformer in experiments/transformer/model.py")
