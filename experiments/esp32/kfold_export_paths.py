"""Resolve k-fold compression checkpoints for ESP32 TFLite export."""

from __future__ import annotations

from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[2]

sys_path = RESEARCH_ROOT
import sys

sys.path.insert(0, str(sys_path))

from experiments.kfold_manifest import (  # noqa: E402
    baseline_run_dir,
    compression_run_dir,
    split_file_for_fold,
)

DEFAULT_MODEL = "tcn"
DEFAULT_FOLD = 0
DEFAULT_SEED = 42


def compression_checkpoint(
    config: str,
    *,
    model: str = DEFAULT_MODEL,
    fold: int = DEFAULT_FOLD,
    seed: int = DEFAULT_SEED,
) -> Path:
    path = compression_run_dir(model, fold, seed, config) / "model.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run k-fold compression first "
            f"(fold={fold}, seed={seed}, config={config})."
        )
    return path


def baseline_checkpoint(
    *,
    model: str = DEFAULT_MODEL,
    fold: int = DEFAULT_FOLD,
    seed: int = DEFAULT_SEED,
) -> Path:
    path = baseline_run_dir(model, fold, seed) / "best_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline checkpoint: {path}")
    return path


def baseline_norm_stats(
    *,
    model: str = DEFAULT_MODEL,
    fold: int = DEFAULT_FOLD,
    seed: int = DEFAULT_SEED,
) -> Path:
    path = baseline_run_dir(model, fold, seed) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing norm_stats.json: {path}")
    return path


def fold_split_file(fold: int = DEFAULT_FOLD) -> Path:
    return split_file_for_fold(fold)
