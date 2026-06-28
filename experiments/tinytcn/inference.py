#!/usr/bin/env python3
"""Run TinyTCN inference on one trial CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXP_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = EXP_DIR.parents[1]
SHARED = RESEARCH_ROOT / "shared"
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(EXP_DIR))

from gait_labels import IMU_CHANNEL_SETS, IMU_INPUT_COLUMNS, PHASE_NAMES, resolve_label_column  # noqa: E402
from data.dataset import NormStats, _clean_imu, make_windows_from_trial  # noqa: E402
from model import TinyTCN  # noqa: E402
from paths import TINYTCN_RUNS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=TINYTCN_RUNS / "fp32_100hz" / "best_model.pt")
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else
        ("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    )

    ckpt = torch.load(args.checkpoint, weights_only=False)
    cfg = ckpt["config"]
    norm = NormStats(**ckpt["norm_stats"])
    feature_columns = cfg.get("feature_columns") or IMU_CHANNEL_SETS.get(
        cfg.get("channels", "bilateral"), IMU_INPUT_COLUMNS
    )
    label_column = cfg.get("label_column") or resolve_label_column(
        pd.read_csv(args.file, nrows=0).columns.tolist(), cfg.get("channels", "bilateral")
    )

    model = TinyTCN(cfg["n_channels"], cfg["n_classes"], cfg["hidden"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    df = pd.read_csv(args.file, usecols=[*feature_columns, label_column])
    x = _clean_imu(df[feature_columns].to_numpy(dtype=np.float32))
    y_true = df[label_column].to_numpy(dtype=np.int64)
    batch = make_windows_from_trial(x, y_true, window_size=cfg["window"], stride=1, norm=norm)
    assert batch is not None

    preds: list[int] = []
    with torch.no_grad():
        for i in range(0, len(batch.y), 512):
            xb = torch.from_numpy(batch.x[i : i + 512]).to(device)
            preds.extend(model(xb).argmax(dim=1).cpu().tolist())

    acc = np.mean(np.array(preds) == batch.y)
    print(f"File:     {args.file.name}")
    print(f"Windows:  {len(batch.y):,}")
    print(f"Accuracy: {acc:.4f}")
    for code in range(4):
        mask = batch.y == code
        if mask.any():
            c_acc = np.mean(np.array(preds)[mask] == batch.y[mask])
            print(f"  {PHASE_NAMES[code]:4s}: {c_acc:.4f}  (n={mask.sum():,})")


if __name__ == "__main__":
    main()
