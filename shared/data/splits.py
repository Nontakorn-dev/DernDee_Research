"""Subject-level train / val / test splits for NONAN GaitPrint."""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

SUBJECT_RE = re.compile(r"^S\d{3}$")


def discover_subjects(data_root: Path) -> list[str]:
    return sorted(
        d.name for d in data_root.iterdir()
        if d.is_dir() and SUBJECT_RE.match(d.name)
    )


def discover_trial_files(data_root: Path, subjects: list[str]) -> list[Path]:
    files: list[Path] = []
    for subj in subjects:
        subj_dir = data_root / subj
        files.extend(sorted(subj_dir.glob("S*.csv")))
    return files


def make_subject_split(
    subjects: list[str],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Split by subject — no leakage across train/val/test."""
    subjects = sorted(subjects)
    rng = np.random.default_rng(seed)
    order = subjects.copy()
    rng.shuffle(order)

    n = len(order)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = n - n_val - n_test

    rows = []
    for subj in order[:n_train]:
        rows.append({"subject": subj, "split": "train"})
    for subj in order[n_train : n_train + n_val]:
        rows.append({"subject": subj, "split": "val"})
    for subj in order[n_train + n_val :]:
        rows.append({"subject": subj, "split": "test"})

    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def save_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    by_split = {
        split: df.loc[df["split"] == split, "subject"].tolist()
        for split in ("train", "val", "test")
    }
    path.with_suffix(".json").write_text(json.dumps(by_split, indent=2))


def load_split(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path)
    return {
        split: df.loc[df["split"] == split, "subject"].tolist()
        for split in ("train", "val", "test")
    }


def files_for_split(data_root: Path, split_map: dict[str, list[str]], key: str) -> list[Path]:
    return discover_trial_files(data_root, split_map[key])


def subjects_from_files(files: list[Path]) -> set[str]:
    """Extract S### subject IDs from trial file paths."""
    return {f.parent.name for f in files if SUBJECT_RE.match(f.parent.name)}


class SplitPolicy(str, Enum):
    TRAIN_ONLY = "train_only"
    VAL_ONLY = "val_only"
    TEST_ONLY = "test_only"


def assert_files_match_split(
    files: list[Path],
    split_map: dict[str, list[str]],
    *,
    policy: SplitPolicy,
    context: str,
) -> None:
    """Runtime check: file subjects match allowed split; zero forbidden overlap."""
    file_subjects = subjects_from_files(files)
    if not file_subjects:
        raise ValueError(f"{context}: no subjects resolved from {len(files)} files")

    train_subjects = set(split_map["train"])
    val_subjects = set(split_map["val"])
    test_subjects = set(split_map["test"])

    if policy == SplitPolicy.TRAIN_ONLY:
        allowed = train_subjects
        forbidden = val_subjects | test_subjects
        if not file_subjects <= allowed:
            raise AssertionError(
                f"{context}: expected train-only, got extra subjects "
                f"{sorted(file_subjects - allowed)}"
            )
    elif policy == SplitPolicy.VAL_ONLY:
        allowed = val_subjects
        forbidden = train_subjects | test_subjects
        if not file_subjects <= allowed:
            raise AssertionError(
                f"{context}: expected val-only, got extra subjects "
                f"{sorted(file_subjects - allowed)}"
            )
    elif policy == SplitPolicy.TEST_ONLY:
        allowed = test_subjects
        forbidden = train_subjects | val_subjects
        if not file_subjects <= allowed:
            raise AssertionError(
                f"{context}: expected test-only, got extra subjects "
                f"{sorted(file_subjects - allowed)}"
            )
    else:
        raise ValueError(f"Unknown split policy: {policy!r}")

    leak = file_subjects & forbidden
    if leak:
        raise AssertionError(
            f"{context}: {policy.value} data includes forbidden subjects {sorted(leak)}"
        )
