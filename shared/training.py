"""Shared training utilities — data loaders and device helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import (
    GaitWindowDataset,
    LazyGaitWindowDataset,
    build_window_arrays,
    build_window_index,
)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def resolve_hz_params(
    *,
    source_hz: int,
    target_hz: int,
    window: int | None,
    stride: int | None,
) -> tuple[int, int, int]:
    if source_hz % target_hz != 0:
        raise ValueError(f"source_hz {source_hz} must divide target_hz {target_hz}")
    decimate = source_hz // target_hz
    win = window or int(0.5 * target_hz)
    st = stride or max(1, int(0.05 * target_hz))
    return decimate, win, st


def make_loader(
    files: list[Path],
    *,
    feature_columns: list[str],
    label_column: str,
    window_size: int,
    stride: int,
    norm,
    decimate: int,
    batch_size: int,
    shuffle: bool,
    eager: bool,
    cache_trials: int,
) -> tuple[DataLoader, np.ndarray]:
    if eager:
        data = build_window_arrays(
            files,
            feature_columns=feature_columns,
            label_column=label_column,
            window_size=window_size,
            stride=stride,
            norm=norm,
            decimate=decimate,
        )
        ds = GaitWindowDataset(data.x, data.y)
        labels = data.y
    else:
        indices = build_window_index(
            files,
            feature_columns=feature_columns,
            label_column=label_column,
            window_size=window_size,
            stride=stride,
            decimate=decimate,
        )
        labels = np.array([wi.label for wi in indices], dtype=np.int64)
        ds = LazyGaitWindowDataset(
            files,
            indices,
            feature_columns=feature_columns,
            label_column=label_column,
            window_size=window_size,
            norm=norm,
            decimate=decimate,
            cache_trials=cache_trials,
        )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, labels
