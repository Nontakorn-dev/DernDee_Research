"""Sliding-window dataset for TinyTCN gait phase classification."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gait_labels import (
    IMU_INPUT_COLUMNS,
    PHASE_UNK,
    label_column_for_channels,
    resolve_label_column,
)  # noqa: E402

N_CLASSES = 4


@dataclass
class NormStats:
    mean: list[float]
    std: list[float]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> NormStats:
        return cls(**json.loads(path.read_text()))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        return (x - mean) / std


@dataclass
class WindowIndex:
    file_idx: int
    end_idx: int
    label: int


@dataclass
class WindowBatch:
    x: np.ndarray  # (N, window, n_features)
    y: np.ndarray  # (N,)
    trial_ids: np.ndarray  # (N,)


@dataclass
class DualWindowBatch:
    x: np.ndarray  # (N, window, n_features)
    y_lt: np.ndarray  # (N,)
    y_rt: np.ndarray  # (N,)
    trial_ids: np.ndarray  # (N,)


@dataclass
class DualWindowIndex:
    file_idx: int
    end_idx: int
    label_lt: int
    label_rt: int


SOURCE_HZ = 200  # processed/Xy is stored at 200 Hz (Noraxon native)


def list_all_trial_files(data_root: Path, subjects: list[str] | None = None) -> list[Path]:
    from data.splits import discover_subjects, discover_trial_files  # noqa: WPS433

    if subjects is None:
        subjects = discover_subjects(data_root)
    return discover_trial_files(data_root, subjects)


def _clean_imu(x: np.ndarray) -> np.ndarray:
    """Fill sparse IMU NaNs via linear interpolation along time."""
    if not np.isnan(x).any():
        return x
    df = pd.DataFrame(x)
    df = df.interpolate(method="linear", limit_direction="both")
    df = df.fillna(0.0)
    return df.to_numpy(dtype=np.float32)


def probe_label_column(files: list[Path], channels: str) -> str:
    """Detect label column from first trial (validates all trials share schema)."""
    sample = pd.read_csv(files[0], nrows=0).columns.tolist()
    return resolve_label_column(sample, channels)


def probe_dual_label_columns(files: list[Path]) -> tuple[str, str]:
    """Detect left + right label columns from first trial."""
    from gait_labels import resolve_dual_label_columns  # noqa: WPS433

    sample = pd.read_csv(files[0], nrows=0).columns.tolist()
    return resolve_dual_label_columns(sample)


def load_trial(
    path: Path,
    *,
    feature_columns: list[str] = IMU_INPUT_COLUMNS,
    label_column: str = "phase",
    decimate: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    cols = [*feature_columns, label_column]
    df = pd.read_csv(path, usecols=cols)
    x = _clean_imu(df[feature_columns].to_numpy(dtype=np.float32))
    y = df[label_column].to_numpy(dtype=np.int64)
    if decimate > 1:
        x = x[::decimate]
        y = y[::decimate]
    return x, y


def load_trial_dual(
    path: Path,
    *,
    feature_columns: list[str] = IMU_INPUT_COLUMNS,
    label_lt_column: str = "phase_lt",
    label_rt_column: str = "phase_rt",
    decimate: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = [*feature_columns, label_lt_column, label_rt_column]
    df = pd.read_csv(path, usecols=cols)
    x = _clean_imu(df[feature_columns].to_numpy(dtype=np.float32))
    y_lt = df[label_lt_column].to_numpy(dtype=np.int64)
    y_rt = df[label_rt_column].to_numpy(dtype=np.int64)
    if decimate > 1:
        x = x[::decimate]
        y_lt = y_lt[::decimate]
        y_rt = y_rt[::decimate]
    return x, y_lt, y_rt


def make_dual_windows_from_trial(
    x: np.ndarray,
    y_lt: np.ndarray,
    y_rt: np.ndarray,
    *,
    window_size: int,
    stride: int,
    norm: NormStats | None = None,
) -> DualWindowBatch | None:
    if norm is not None:
        x = norm.normalize(x)

    n = len(x)
    if n < window_size:
        return None

    n_features = x.shape[1]
    indices: list[int] = []
    for end in range(window_size - 1, n, stride):
        if y_lt[end] == PHASE_UNK or y_rt[end] == PHASE_UNK:
            continue
        indices.append(end)

    if not indices:
        return None

    n_win = len(indices)
    windows = np.empty((n_win, window_size, n_features), dtype=np.float32)
    labels_lt = np.empty(n_win, dtype=np.int64)
    labels_rt = np.empty(n_win, dtype=np.int64)

    for i, end in enumerate(indices):
        start = end - window_size + 1
        windows[i] = x[start : end + 1]
        labels_lt[i] = y_lt[end]
        labels_rt[i] = y_rt[end]

    return DualWindowBatch(
        x=windows,
        y_lt=labels_lt,
        y_rt=labels_rt,
        trial_ids=np.full(n_win, -1, dtype=np.int64),
    )


def build_dual_window_arrays(
    files: list[Path],
    *,
    feature_columns: list[str] = IMU_INPUT_COLUMNS,
    label_lt_column: str = "phase_lt",
    label_rt_column: str = "phase_rt",
    window_size: int,
    stride: int,
    norm: NormStats | None = None,
    decimate: int = 1,
) -> DualWindowBatch:
    xs: list[np.ndarray] = []
    y_lts: list[np.ndarray] = []
    y_rts: list[np.ndarray] = []
    tids: list[np.ndarray] = []

    for trial_id, path in enumerate(tqdm(files, desc="build dual windows", unit="trial")):
        x, y_lt, y_rt = load_trial_dual(
            path,
            feature_columns=feature_columns,
            label_lt_column=label_lt_column,
            label_rt_column=label_rt_column,
            decimate=decimate,
        )
        batch = make_dual_windows_from_trial(
            x, y_lt, y_rt, window_size=window_size, stride=stride, norm=norm
        )
        if batch is None:
            continue
        batch.trial_ids[:] = trial_id
        xs.append(batch.x)
        y_lts.append(batch.y_lt)
        y_rts.append(batch.y_rt)
        tids.append(batch.trial_ids)

    if not xs:
        raise ValueError("No dual windows produced — check data and parameters.")

    return DualWindowBatch(
        x=np.concatenate(xs, axis=0),
        y_lt=np.concatenate(y_lts, axis=0),
        y_rt=np.concatenate(y_rts, axis=0),
        trial_ids=np.concatenate(tids, axis=0),
    )


def build_dual_window_index(
    files: list[Path],
    *,
    feature_columns: list[str] = IMU_INPUT_COLUMNS,
    label_lt_column: str = "phase_lt",
    label_rt_column: str = "phase_rt",
    window_size: int,
    stride: int,
    decimate: int = 1,
) -> list[DualWindowIndex]:
    indices: list[DualWindowIndex] = []
    for file_idx, path in enumerate(tqdm(files, desc="index dual windows", unit="trial")):
        _, y_lt, y_rt = load_trial_dual(
            path,
            feature_columns=feature_columns,
            label_lt_column=label_lt_column,
            label_rt_column=label_rt_column,
            decimate=decimate,
        )
        n = len(y_lt)
        if n < window_size:
            continue
        for end in range(window_size - 1, n, stride):
            label_lt = int(y_lt[end])
            label_rt = int(y_rt[end])
            if label_lt == PHASE_UNK or label_rt == PHASE_UNK:
                continue
            indices.append(
                DualWindowIndex(file_idx=file_idx, end_idx=end, label_lt=label_lt, label_rt=label_rt)
            )
    if not indices:
        raise ValueError("No dual windows produced — check data and parameters.")
    return indices


def fit_norm_stats(
    train_files: list[Path],
    *,
    feature_columns: list[str] = IMU_INPUT_COLUMNS,
    decimate: int = 1,
) -> NormStats:
    chunks: list[np.ndarray] = []
    for path in tqdm(train_files, desc="fit norm stats", unit="trial"):
        x, _ = load_trial(path, feature_columns=feature_columns, decimate=decimate)
        chunks.append(x)
    all_x = np.concatenate(chunks, axis=0)
    mean = np.nanmean(all_x, axis=0)
    std = np.nanstd(all_x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return NormStats(mean=mean.tolist(), std=std.tolist())


def build_window_index(
    files: list[Path],
    *,
    feature_columns: list[str] = IMU_INPUT_COLUMNS,
    label_column: str = "phase",
    window_size: int,
    stride: int,
    decimate: int = 1,
) -> list[WindowIndex]:
    """Lightweight window index — stores (file, end, label) only, not window data."""
    indices: list[WindowIndex] = []
    for file_idx, path in enumerate(tqdm(files, desc="index windows", unit="trial")):
        _, y = load_trial(
            path, feature_columns=feature_columns, label_column=label_column, decimate=decimate
        )
        n = len(y)
        if n < window_size:
            continue
        for end in range(window_size - 1, n, stride):
            label = int(y[end])
            if label == PHASE_UNK:
                continue
            indices.append(WindowIndex(file_idx=file_idx, end_idx=end, label=label))
    if not indices:
        raise ValueError("No windows produced — check data and parameters.")
    return indices


def make_windows_from_trial(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window_size: int,
    stride: int,
    norm: NormStats | None = None,
) -> WindowBatch | None:
    """Causal windows ending at index t; label = phase[t]."""
    if norm is not None:
        x = norm.normalize(x)

    n = len(x)
    if n < window_size:
        return None

    n_features = x.shape[1]
    indices: list[int] = []
    for end in range(window_size - 1, n, stride):
        if y[end] == PHASE_UNK:
            continue
        indices.append(end)

    if not indices:
        return None

    n_win = len(indices)
    windows = np.empty((n_win, window_size, n_features), dtype=np.float32)
    labels = np.empty(n_win, dtype=np.int64)

    for i, end in enumerate(indices):
        start = end - window_size + 1
        windows[i] = x[start : end + 1]
        labels[i] = y[end]

    return WindowBatch(x=windows, y=labels, trial_ids=np.full(n_win, -1, dtype=np.int64))


def build_window_arrays(
    files: list[Path],
    *,
    feature_columns: list[str] = IMU_INPUT_COLUMNS,
    label_column: str = "phase",
    window_size: int,
    stride: int,
    norm: NormStats | None = None,
    decimate: int = 1,
) -> WindowBatch:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    tids: list[np.ndarray] = []

    for trial_id, path in enumerate(tqdm(files, desc="build windows", unit="trial")):
        x, y = load_trial(
            path, feature_columns=feature_columns, label_column=label_column, decimate=decimate
        )
        batch = make_windows_from_trial(
            x, y, window_size=window_size, stride=stride, norm=norm
        )
        if batch is None:
            continue
        batch.trial_ids[:] = trial_id
        xs.append(batch.x)
        ys.append(batch.y)
        tids.append(batch.trial_ids)

    if not xs:
        raise ValueError("No windows produced — check data and parameters.")

    return WindowBatch(
        x=np.concatenate(xs, axis=0),
        y=np.concatenate(ys, axis=0),
        trial_ids=np.concatenate(tids, axis=0),
    )


class GaitWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class DualGaitWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y_lt: np.ndarray, y_rt: np.ndarray) -> None:
        self.x = torch.from_numpy(x)
        self.y_lt = torch.from_numpy(y_lt)
        self.y_rt = torch.from_numpy(y_rt)

    def __len__(self) -> int:
        return len(self.y_lt)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y_lt[idx], self.y_rt[idx]


class LazyDualGaitWindowDataset(Dataset):
    """Lazy dual-label windows — low RAM."""

    def __init__(
        self,
        files: list[Path],
        indices: list[DualWindowIndex],
        *,
        feature_columns: list[str],
        label_lt_column: str,
        label_rt_column: str,
        window_size: int,
        norm: NormStats,
        decimate: int = 1,
        cache_trials: int = 3,
    ) -> None:
        self.files = files
        self.indices = indices
        self.feature_columns = feature_columns
        self.label_lt_column = label_lt_column
        self.label_rt_column = label_rt_column
        self.window_size = window_size
        self.norm = norm
        self.decimate = decimate
        self.cache_trials = cache_trials
        self._cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []

    def _trial_x(self, file_idx: int) -> np.ndarray:
        if file_idx not in self._cache:
            x, _, _ = load_trial_dual(
                self.files[file_idx],
                feature_columns=self.feature_columns,
                label_lt_column=self.label_lt_column,
                label_rt_column=self.label_rt_column,
                decimate=self.decimate,
            )
            x = self.norm.normalize(x)
            if len(self._cache) >= self.cache_trials:
                evict = self._cache_order.pop(0)
                del self._cache[evict]
            self._cache[file_idx] = x
            self._cache_order.append(file_idx)
        return self._cache[file_idx]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wi = self.indices[idx]
        x = self._trial_x(wi.file_idx)
        start = wi.end_idx - self.window_size + 1
        window = x[start : wi.end_idx + 1]
        return (
            torch.from_numpy(window.copy()),
            torch.tensor(wi.label_lt, dtype=torch.long),
            torch.tensor(wi.label_rt, dtype=torch.long),
        )


class LazyGaitWindowDataset(Dataset):
    """Load trials on demand — low RAM, suitable for Colab."""

    def __init__(
        self,
        files: list[Path],
        indices: list[WindowIndex],
        *,
        feature_columns: list[str],
        label_column: str = "phase",
        window_size: int,
        norm: NormStats,
        decimate: int = 1,
        cache_trials: int = 3,
    ) -> None:
        self.files = files
        self.indices = indices
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.window_size = window_size
        self.norm = norm
        self.decimate = decimate
        self.cache_trials = cache_trials
        self._cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []

    def _trial_x(self, file_idx: int) -> np.ndarray:
        if file_idx not in self._cache:
            x, _ = load_trial(
                self.files[file_idx],
                feature_columns=self.feature_columns,
                label_column=self.label_column,
                decimate=self.decimate,
            )
            x = self.norm.normalize(x)
            if len(self._cache) >= self.cache_trials:
                evict = self._cache_order.pop(0)
                del self._cache[evict]
            self._cache[file_idx] = x
            self._cache_order.append(file_idx)
        return self._cache[file_idx]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wi = self.indices[idx]
        x = self._trial_x(wi.file_idx)
        start = wi.end_idx - self.window_size + 1
        window = x[start : wi.end_idx + 1]
        return torch.from_numpy(window.copy()), torch.tensor(wi.label, dtype=torch.long)

