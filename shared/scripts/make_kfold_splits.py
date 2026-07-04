#!/usr/bin/env python3
"""Generate stratified k-fold subject splits with rotating validation per fold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
SHARED = RESEARCH_ROOT / "shared"
sys.path.insert(0, str(SHARED))

from data.splits import discover_subjects, discover_trial_files, save_split  # noqa: E402
from gait_labels import resolve_label_column  # noqa: E402

PHASE_TO_IDX = {"LR": 0, "LS": 1, "PSw": 2, "Sw": 3}
MINORITY_PHASES = (0, 2)  # LR + PSw


def _phase_indices(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("Int64")
    labels = series.astype(str).str.strip().str.upper()
    return labels.map(PHASE_TO_IDX)


def minority_phase_rate(data_root: Path, subject: str, *, channels: str = "bilateral") -> float:
    """Fraction of labeled windows that are LR or PSw for one subject."""
    files = discover_trial_files(data_root, [subject])
    if not files:
        return 0.0

    total = 0
    minority = 0
    for path in files:
        df = pd.read_csv(path, usecols=lambda c: c in ("phase_lt", "phase_rt", "phase"))
        if df.empty:
            continue
        label_col = resolve_label_column(list(df.columns), channels)
        mapped = _phase_indices(df[label_col])
        valid = mapped.notna() & (mapped >= 0)
        total += int(valid.sum())
        minority += int(mapped[valid].isin(MINORITY_PHASES).sum())

    return minority / total if total else 0.0


def stratify_bins(rates: np.ndarray, *, n_bins: int = 5) -> np.ndarray:
    """Bin continuous minority rates for StratifiedKFold."""
    if len(np.unique(rates)) <= 1:
        return np.zeros(len(rates), dtype=int)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(rates, quantiles)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros(len(rates), dtype=int)
    return np.digitize(rates, edges[1:-1], right=True)


def assign_val_subjects(
    pool: list[str],
    rates: dict[str, float],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Stratified val sample from non-test pool; remainder is train."""
    pool = sorted(pool)
    if not pool:
        raise ValueError("Empty subject pool for val assignment")

    n_val = max(1, int(round(len(pool) * val_ratio)))
    n_val = min(n_val, len(pool) - 1)

    pool_rates = np.array([rates[s] for s in pool])
    bins = stratify_bins(pool_rates)
    rng = np.random.default_rng(seed)

    val: list[str] = []
    remaining = pool.copy()
    remaining_bins = {s: b for s, b in zip(pool, bins)}

    by_bin: dict[int, list[str]] = {}
    for subj in pool:
        by_bin.setdefault(remaining_bins[subj], []).append(subj)

    while len(val) < n_val and remaining:
        for bin_id in sorted(by_bin):
            candidates = [s for s in by_bin[bin_id] if s in remaining]
            if not candidates:
                continue
            pick = candidates[rng.integers(0, len(candidates))]
            val.append(pick)
            remaining.remove(pick)
            if len(val) >= n_val:
                break

    if len(val) < n_val:
        rng.shuffle(remaining)
        val.extend(remaining[: n_val - len(val)])
        remaining = remaining[n_val - len(val) :]

    train = sorted(set(pool) - set(val))
    return train, sorted(val)


def build_fold_splits(
    subjects: list[str],
    rates: dict[str, float],
    *,
    n_folds: int,
    val_ratio: float,
    base_seed: int,
) -> list[dict[str, object]]:
    subject_arr = np.array(sorted(subjects))
    rate_arr = np.array([rates[s] for s in subject_arr])
    strata = stratify_bins(rate_arr)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=base_seed)
    folds: list[dict[str, object]] = []

    for fold_idx, (_, test_idx) in enumerate(skf.split(subject_arr, strata)):
        test_subjects = sorted(subject_arr[test_idx].tolist())
        pool = sorted(set(subject_arr.tolist()) - set(test_subjects))
        fold_seed = base_seed + fold_idx
        train_subjects, val_subjects = assign_val_subjects(
            pool, rates, val_ratio=val_ratio, seed=fold_seed
        )

        rows = []
        for subj in train_subjects:
            rows.append({"subject": subj, "split": "train"})
        for subj in val_subjects:
            rows.append({"subject": subj, "split": "val"})
        for subj in test_subjects:
            rows.append({"subject": subj, "split": "test"})

        folds.append(
            {
                "fold": fold_idx,
                "dataframe": pd.DataFrame(rows).sort_values("subject").reset_index(drop=True),
                "meta": {
                    "fold": fold_idx,
                    "val_rotation": True,
                    "base_seed": base_seed,
                    "fold_seed": fold_seed,
                    "train_subjects": train_subjects,
                    "val_subjects": val_subjects,
                    "test_subjects": test_subjects,
                },
            }
        )

    return folds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=RESEARCH_ROOT / "dataset" / "Xy")
    p.add_argument("--output-dir", type=Path, default=SHARED / "splits" / "folds")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--channels", type=str, default="bilateral")
    args = p.parse_args()

    subjects = discover_subjects(args.data_root)
    print(f"Discovered {len(subjects)} subjects under {args.data_root}")

    rates = {s: minority_phase_rate(args.data_root, s, channels=args.channels) for s in subjects}
    folds = build_fold_splits(
        subjects,
        rates,
        n_folds=args.n_folds,
        val_ratio=args.val_ratio,
        base_seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"n_folds": args.n_folds, "val_ratio": args.val_ratio, "seed": args.seed, "folds": []}

    for fold in folds:
        fold_idx = int(fold["fold"])
        csv_path = args.output_dir / f"fold{fold_idx}.csv"
        save_split(fold["dataframe"], csv_path)
        meta_path = args.output_dir / f"fold{fold_idx}_meta.json"
        meta_path.write_text(json.dumps(fold["meta"], indent=2))

        meta = fold["meta"]
        print(
            f"fold{fold_idx}: train={len(meta['train_subjects'])} "
            f"val={len(meta['val_subjects'])} test={len(meta['test_subjects'])}"
        )
        summary["folds"].append(meta)

    (args.output_dir / "kfold_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {len(folds)} folds to {args.output_dir}")


if __name__ == "__main__":
    main()
