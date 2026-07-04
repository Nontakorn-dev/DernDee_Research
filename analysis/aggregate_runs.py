#!/usr/bin/env python3
"""Aggregate k-fold × seed metrics with variance decomposition and paired tests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
PHASES = ("LR", "LS", "PSw", "Sw")
BASELINE_MODELS = ("tinytcn", "cnn1d", "tcn", "lstm_gru", "transformer")
COMPRESSION_CONFIGS = ("FP32", "INT8", "Prune50", "INT8+Prune50")
DEFAULT_SEEDS = (42, 123, 456)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def baseline_run_dir(model: str, fold: int, seed: int) -> Path:
    return RESEARCH_ROOT / "experiments" / model / "runs" / f"fold{fold}_seed{seed}_fp32_100hz"


def compression_run_dir(fold: int, seed: int, config: str) -> Path:
    return RESEARCH_ROOT / "experiments" / "compression" / "runs" / f"fold{fold}_seed{seed}" / config


def read_metrics(path: Path) -> dict[str, Any] | None:
    metrics_path = path / "test_metrics.json"
    if not metrics_path.exists():
        metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return None
    return load_json(metrics_path)


def metric_value(data: dict[str, Any], metric: str, *, phase: str | None = None) -> float | None:
    if phase is None:
        value = data.get(metric)
    else:
        value = data.get(metric, {}).get(phase)
    return None if value is None else float(value)


def collect_scores(
    run_dir_fn,
    *,
    labels: tuple[str, ...],
    n_folds: int,
    seeds: tuple[int, ...],
    metric: str,
    phase: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return (n_labels, n_folds, n_seeds) array and list of missing runs."""
    scores = np.full((len(labels), n_folds, len(seeds)), np.nan)
    missing: list[str] = []

    for li, label in enumerate(labels):
        for fi in range(n_folds):
            for si, seed in enumerate(seeds):
                run_dir = run_dir_fn(label, fi, seed)
                data = read_metrics(run_dir)
                value = metric_value(data, metric, phase=phase) if data is not None else None
                if value is None:
                    missing.append(str(run_dir))
                else:
                    scores[li, fi, si] = value

    return scores, missing


def axis_nanmean(scores: np.ndarray, axis: int) -> np.ndarray:
    counts = np.sum(~np.isnan(scores), axis=axis)
    sums = np.nansum(scores, axis=axis)
    out = np.full(sums.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def nanstd(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return math.nan
    return float(np.std(valid))


def variance_decomposition(scores: np.ndarray) -> dict[str, float]:
    """scores shape (n_folds, n_seeds)."""
    valid = scores[~np.isnan(scores)]
    if valid.size == 0:
        return {
            "overall_mean": math.nan,
            "overall_std": math.nan,
            "std_across_folds": math.nan,
            "std_across_seeds": math.nan,
        }

    fold_means = axis_nanmean(scores, axis=1)
    seed_means = axis_nanmean(scores, axis=0)
    return {
        "overall_mean": float(np.nanmean(scores)),
        "overall_std": nanstd(scores),
        "std_across_folds": nanstd(fold_means),
        "std_across_seeds": nanstd(seed_means),
    }


def summarize_scores(scores: np.ndarray) -> dict[str, Any]:
    entry = variance_decomposition(scores)
    entry["scores"] = scores.tolist()
    return entry


def paired_test(a: np.ndarray, b: np.ndarray) -> dict[str, float | str | None]:
    """Paired comparison on flattened fold×seed pairs (drop NaN pairs)."""
    mask = ~(np.isnan(a) | np.isnan(b))
    x = a[mask]
    y = b[mask]
    if len(x) < 2:
        return {"n_pairs": len(x), "wilcoxon_p": None, "ttest_p": None, "mean_diff": None}

    try:
        wilcoxon = stats.wilcoxon(x, y, zero_method="wilcox")
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_p = None

    ttest = stats.ttest_rel(x, y, nan_policy="omit")
    return {
        "n_pairs": int(len(x)),
        "mean_diff": float(np.mean(x - y)),
        "wilcoxon_p": wilcoxon_p,
        "ttest_p": float(ttest.pvalue) if not np.isnan(ttest.pvalue) else None,
    }


def aggregate_baselines(*, n_folds: int, seeds: tuple[int, ...]) -> dict[str, Any]:
    scores, missing = collect_scores(
        lambda model, fold, seed: baseline_run_dir(model, fold, seed),
        labels=BASELINE_MODELS,
        n_folds=n_folds,
        seeds=seeds,
        metric="macro_f1",
    )

    phase_scores: dict[str, np.ndarray] = {}
    all_missing = set(missing)
    for phase in PHASES:
        phase_scores[phase], phase_missing = collect_scores(
            lambda model, fold, seed: baseline_run_dir(model, fold, seed),
            labels=BASELINE_MODELS,
            n_folds=n_folds,
            seeds=seeds,
            metric="phase_f1",
            phase=phase,
        )
        all_missing.update(phase_missing)

    results: dict[str, Any] = {"models": {}, "missing_runs": sorted(all_missing)}
    for mi, model in enumerate(BASELINE_MODELS):
        results["models"][model] = {
            "macro_f1": summarize_scores(scores[mi]),
            "phase_f1": {
                phase: summarize_scores(phase_scores[phase][mi])
                for phase in PHASES
            },
        }

    ref = scores[BASELINE_MODELS.index("tinytcn")]
    comparisons: dict[str, Any] = {}
    for mi, model in enumerate(BASELINE_MODELS):
        if model == "tinytcn":
            continue
        comparisons[model] = paired_test(ref, scores[mi])
    results["tinytcn_vs_baselines"] = comparisons
    return results


def aggregate_compression(*, n_folds: int, seeds: tuple[int, ...]) -> dict[str, Any]:
    scores, missing = collect_scores(
        lambda config, fold, seed: compression_run_dir(fold, seed, config),
        labels=COMPRESSION_CONFIGS,
        n_folds=n_folds,
        seeds=seeds,
        metric="macro_f1",
    )

    phase_scores: dict[str, np.ndarray] = {}
    all_missing = set(missing)
    for phase in PHASES:
        phase_scores[phase], phase_missing = collect_scores(
            lambda config, fold, seed: compression_run_dir(fold, seed, config),
            labels=COMPRESSION_CONFIGS,
            n_folds=n_folds,
            seeds=seeds,
            metric="phase_f1",
            phase=phase,
        )
        all_missing.update(phase_missing)

    results: dict[str, Any] = {"configs": {}, "missing_runs": sorted(all_missing)}
    for ci, config in enumerate(COMPRESSION_CONFIGS):
        results["configs"][config] = {
            "macro_f1": summarize_scores(scores[ci]),
            "phase_f1": {
                phase: summarize_scores(phase_scores[phase][ci])
                for phase in PHASES
            },
        }

    ref = scores[COMPRESSION_CONFIGS.index("FP32")]
    comparisons: dict[str, Any] = {}
    for ci, config in enumerate(COMPRESSION_CONFIGS):
        if config == "FP32":
            continue
        comparisons[config] = paired_test(ref, scores[ci])
    results["fp32_vs_compression"] = comparisons
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--out", type=Path, default=RESEARCH_ROOT / "analysis" / "aggregated_runs.json")
    args = p.parse_args()

    seeds = tuple(args.seeds)
    payload = {
        "n_folds": args.n_folds,
        "seeds": list(seeds),
        "baselines": aggregate_baselines(n_folds=args.n_folds, seeds=seeds),
        "compression": aggregate_compression(n_folds=args.n_folds, seeds=seeds),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")
    print(f"Missing baseline runs: {len(payload['baselines']['missing_runs'])}")
    print(f"Missing compression runs: {len(payload['compression']['missing_runs'])}")


if __name__ == "__main__":
    main()
