#!/usr/bin/env python3
"""QA report for k-fold splits: phase balance, val rotation overlap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SHARED = RESEARCH_ROOT / "shared"
sys.path.insert(0, str(SHARED))

from data.splits import discover_trial_files, load_split  # noqa: E402
from gait_labels import resolve_label_column  # noqa: E402

PHASE_TO_IDX = {"LR": 0, "LS": 1, "PSw": 2, "Sw": 3}
PHASE_NAMES = ("LR", "LS", "PSw", "Sw")


def phase_support_for_subjects(data_root: Path, subjects: list[str], *, channels: str) -> dict[str, int]:
    counts = {p: 0 for p in PHASE_NAMES}
    idx_to_name = {0: "LR", 1: "LS", 2: "PSw", 3: "Sw"}
    for subj in subjects:
        for path in discover_trial_files(data_root, [subj]):
            df = pd.read_csv(path, usecols=lambda c: c in ("phase_lt", "phase_rt", "phase"))
            if df.empty:
                continue
            label_col = resolve_label_column(list(df.columns), channels)
            series = df[label_col]
            if pd.api.types.is_numeric_dtype(series):
                for val in series.dropna().astype(int):
                    if val in idx_to_name:
                        counts[idx_to_name[val]] += 1
            else:
                labels = series.astype(str).str.strip().str.upper()
                for phase in labels:
                    if phase in counts:
                        counts[phase] += 1
    return counts


def val_overlap_matrix(folds_dir: Path, n_folds: int) -> np.ndarray:
    val_sets: list[set[str]] = []
    for i in range(n_folds):
        meta = json.loads((folds_dir / f"fold{i}_meta.json").read_text())
        val_sets.append(set(meta["val_subjects"]))
    mat = np.zeros((n_folds, n_folds), dtype=int)
    for i in range(n_folds):
        for j in range(n_folds):
            mat[i, j] = len(val_sets[i] & val_sets[j])
    return mat


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folds-dir", type=Path, default=SHARED / "splits" / "folds")
    p.add_argument("--data-root", type=Path, default=RESEARCH_ROOT / "dataset" / "Xy")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--ps-w-min-pct", type=float, default=3.0, help="Flag folds below this PSw pct")
    p.add_argument("--out", type=Path, default=RESEARCH_ROOT / "analysis" / "fold_qa_report.json")
    p.add_argument("--channels", type=str, default="bilateral")
    args = p.parse_args()

    report: dict[str, object] = {"folds": [], "val_overlap_matrix": [], "flags": []}
    overlap = val_overlap_matrix(args.folds_dir, args.n_folds)
    report["val_overlap_matrix"] = overlap.tolist()

    for fold_idx in range(args.n_folds):
        split_map = load_split(args.folds_dir / f"fold{fold_idx}.csv")
        fold_entry: dict[str, object] = {"fold": fold_idx, "splits": {}}
        for split_name in ("train", "val", "test"):
            subjects = split_map[split_name]
            counts = phase_support_for_subjects(args.data_root, subjects, channels=args.channels)
            total = sum(counts.values()) or 1
            pct = {p: round(100.0 * counts[p] / total, 2) for p in PHASE_NAMES}
            fold_entry["splits"][split_name] = {
                "n_subjects": len(subjects),
                "subjects": subjects,
                "phase_counts": counts,
                "phase_pct": pct,
            }
            if split_name == "test" and pct["PSw"] < args.ps_w_min_pct:
                report["flags"].append(
                    f"fold{fold_idx} test PSw support {pct['PSw']:.2f}% < {args.ps_w_min_pct}%"
                )

        report["folds"].append(fold_entry)

    off_diag = overlap[~np.eye(args.n_folds, dtype=bool)]
    report["val_overlap_mean_off_diag"] = float(off_diag.mean()) if off_diag.size else 0.0
    report["val_overlap_max_off_diag"] = int(off_diag.max()) if off_diag.size else 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"Val overlap matrix (off-diag mean={report['val_overlap_mean_off_diag']:.2f}):")
    print(overlap)
    if report["flags"]:
        print("\nFlags:")
        for flag in report["flags"]:
            print(f"  - {flag}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
