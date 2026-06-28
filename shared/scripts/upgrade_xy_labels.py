#!/usr/bin/env python3
"""Add phase_lt + phase_rt to processed/Xy trial CSVs.

Priority:
  1. Noraxon contact GT from NONAN_Dataset (if available)
  2. IMU-derived labels from shank gyro (rt_gyr for phase_rt)
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RESEARCH_ROOT / "shared"))

from gait_labels import (  # noqa: E402
    IMU_INPUT_COLUMNS,
    add_gait_labels,
    label_foot_phases,
    phase_labels_from_shank_imu,
)
from preprocess_gaitprint import OUTPUT_COLUMNS  # noqa: E402

SUBJECT_RE = re.compile(r"^S\d{3}$")
SOURCE_HZ = 200


def discover_xy_trials(xy_root: Path) -> list[Path]:
    return sorted(xy_root.rglob("S*.csv"))


def _resolve_phase_lt(df: pd.DataFrame) -> pd.Series:
    if "phase_lt" in df.columns:
        return df["phase_lt"]
    if "phase" in df.columns:
        return df["phase"]
    raise KeyError("No phase_lt or legacy phase column")


def upgrade_trial(
    xy_path: Path,
    xy_root: Path,
    nonan_root: Path | None,
    *,
    hz: float = SOURCE_HZ,
) -> str:
    header = pd.read_csv(xy_path, nrows=0).columns.tolist()
    if "phase_rt" in header and "phase_lt" in header:
        return "skip"

    rel = xy_path.relative_to(xy_root)
    raw_path = nonan_root / rel if nonan_root is not None else None

    if raw_path is not None and raw_path.exists():
        raw = pd.read_csv(raw_path, usecols=OUTPUT_COLUMNS)[OUTPUT_COLUMNS]
        labeled = add_gait_labels(raw)
        xy = pd.read_csv(xy_path)
        if len(labeled) != len(xy):
            return f"row mismatch: {xy_path.name}"
        xy["phase_lt"] = labeled["phase_lt"].to_numpy()
        xy["phase_rt"] = labeled["phase_rt"].to_numpy()
        if "phase" in xy.columns:
            xy = xy.drop(columns=["phase"])
        xy.to_csv(xy_path, index=False)
        return "ok_nonan"

    xy = pd.read_csv(xy_path)
    xy["phase_lt"] = _resolve_phase_lt(xy).to_numpy()

    rt_gyr_y = xy["rt_gyr_y"].to_numpy(dtype=np.float64)
    rt_gyr_x = xy["rt_gyr_x"].to_numpy(dtype=np.float64)
    xy["phase_rt"] = phase_labels_from_shank_imu(rt_gyr_y, rt_gyr_x, hz=hz)

    if "phase" in xy.columns:
        xy = xy.drop(columns=["phase"])

    xy.to_csv(xy_path, index=False)
    return "ok_imu"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xy-root", type=Path, default=RESEARCH_ROOT / "dataset" / "Xy")
    p.add_argument("--nonan-root", type=Path, default=RESEARCH_ROOT / "NONAN_Dataset")
    p.add_argument("--subject", type=str, default=None)
    p.add_argument("--imu-only", action="store_true",
                   help="Skip NONAN even if present; use gyro-derived labels only")
    args = p.parse_args()

    nonan_root = None if args.imu_only or not args.nonan_root.is_dir() else args.nonan_root

    if args.subject:
        trials = discover_xy_trials(args.xy_root / args.subject)
    else:
        trials = discover_xy_trials(args.xy_root)

    if not trials:
        raise SystemExit(f"No trials under {args.xy_root}")

    print(f"XY root:    {args.xy_root}")
    print(f"NONAN:      {nonan_root or '(skipped — IMU fallback)'}")
    print(f"Trials:     {len(trials)}\n")

    counts: dict[str, int] = {}
    t0 = time.time()
    for path in tqdm(trials, desc="upgrade labels", unit="trial"):
        result = upgrade_trial(path, args.xy_root, nonan_root)
        counts[result] = counts.get(result, 0) + 1
        if result.startswith("row") or result.startswith("missing"):
            tqdm.write(f"  FAIL {path.name}: {result}")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")
    for key, n in sorted(counts.items()):
        print(f"  {key}: {n}")


if __name__ == "__main__":
    main()
