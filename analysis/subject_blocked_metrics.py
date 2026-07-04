#!/usr/bin/env python3
"""Per-subject blocked macro F1 with bootstrap CI (subject-level, not window-level)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SHARED = RESEARCH_ROOT / "shared"
sys.path.insert(0, str(SHARED))

from data.dataset import NormStats, make_windows_from_trial, probe_label_column  # noqa: E402
from data.splits import discover_trial_files, load_split  # noqa: E402
from eval_checkpoint import load_checkpoint, model_from_checkpoint  # noqa: E402
from gait_labels import IMU_CHANNEL_SETS  # noqa: E402
from training import pick_device  # noqa: E402

PHASES = ("LR", "LS", "PSw", "Sw")
PHASE_TO_IDX = {"LR": 0, "LS": 1, "PSW": 2, "PSw": 2, "Sw": 3}


def labels_to_indices(series) -> np.ndarray:
    """Convert numeric or string phase labels to model class indices."""
    import pandas as pd

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)

    labels = series.astype(str).str.strip()
    mapped = labels.map(PHASE_TO_IDX)
    missing = mapped.isna()
    if missing.any():
        mapped.loc[missing] = labels.loc[missing].str.upper().map(PHASE_TO_IDX)
    return mapped.fillna(-1).to_numpy(dtype=np.int64)


def predict_subject(
    model: torch.nn.Module,
    device: torch.device,
    trial_path: Path,
    *,
    feature_columns: list[str],
    label_column: str,
    window: int,
    stride: int,
    norm: NormStats,
    decimate: int,
) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(trial_path)
    x = df[feature_columns].to_numpy(dtype=np.float32)
    y = labels_to_indices(df[label_column])

    if decimate > 1:
        x = x[::decimate]
        y = y[::decimate]

    x = norm.normalize(x)
    batch = make_windows_from_trial(x, y, window_size=window, stride=stride)
    if batch is None or len(batch.y) == 0:
        return np.array([]), np.array([])

    model.eval()
    preds: list[int] = []
    with torch.no_grad():
        for i in range(0, len(batch.y), 512):
            xb = torch.tensor(batch.x[i : i + 512], dtype=torch.float32, device=device)
            logits = model(xb)
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    return batch.y, np.array(preds, dtype=np.int64)


def bootstrap_ci(values: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return math.nan, math.nan
    means = []
    for _ in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        means.append(float(np.mean(sample)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_run(
    checkpoint: Path,
    split_file: Path,
    data_root: Path,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    ckpt = load_checkpoint(checkpoint)
    cfg = ckpt["config"]
    model = model_from_checkpoint(ckpt, checkpoint)
    device = pick_device("auto")
    model.to(device)

    norm = NormStats(**ckpt["norm_stats"])
    split_map = load_split(split_file)
    test_subjects = split_map["test"]
    channels = cfg.get("channels", "bilateral")
    feature_columns = cfg.get("feature_columns") or IMU_CHANNEL_SETS[channels]
    label_column = cfg.get("label_column") or probe_label_column(
        discover_trial_files(data_root, test_subjects[:1]), channels
    )

    per_subject: dict[str, float] = {}
    for subj in test_subjects:
        y_true_all: list[int] = []
        y_pred_all: list[int] = []
        for trial in discover_trial_files(data_root, [subj]):
            y_true, y_pred = predict_subject(
                model,
                device,
                trial,
                feature_columns=feature_columns,
                label_column=label_column,
                window=int(cfg["window"]),
                stride=int(cfg.get("test_stride", 1)),
                norm=norm,
                decimate=int(cfg.get("decimate", 1)),
            )
            if len(y_true):
                y_true_all.extend(y_true.tolist())
                y_pred_all.extend(y_pred.tolist())
        if y_true_all:
            per_subject[subj] = float(
                f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
            )

    values = np.array(list(per_subject.values()), dtype=float)
    ci_low, ci_high = bootstrap_ci(values, n_boot=n_boot, seed=seed)
    return {
        "checkpoint": str(checkpoint),
        "split_file": str(split_file),
        "per_subject_macro_f1": per_subject,
        "mean_macro_f1": float(np.mean(values)) if len(values) else None,
        "std_macro_f1": float(np.std(values)) if len(values) else None,
        "bootstrap_ci_95": [ci_low, ci_high],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=RESEARCH_ROOT / "dataset" / "Xy")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    result = evaluate_run(
        args.checkpoint,
        args.split_file,
        args.data_root,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
