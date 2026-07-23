#!/usr/bin/env python3
"""Aggregate per-(config, fold, seed) real-TFLite accuracy results into N=15 stats.

Reads analysis/tflite_real_accuracy_results/*.json (written by
tflite_real_accuracy_batch.py) and pairs them against the PyTorch FP32 and
QDQ-proxy scores already in analysis/aggregated_runs.json (same fold/seed
indexing) to compute mean/std and paired Wilcoxon signed-rank tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = RESEARCH_ROOT / "analysis" / "tflite_real_accuracy_results"
AGG_PATH = RESEARCH_ROOT / "analysis" / "aggregated_runs.json"

SEEDS = [42, 123, 456]
FOLDS = [0, 1, 2, 3, 4]
CONFIGS = ["INT8", "INT8+Prune50"]


def load_real_tflite(config: str) -> dict[tuple[int, int], dict]:
    tag = config.replace("+", "_plus_")
    out = {}
    for seed in SEEDS:
        for fold in FOLDS:
            path = RESULTS_DIR / f"{tag}_fold{fold}_seed{seed}.json"
            if not path.exists():
                continue
            out[(fold, seed)] = json.loads(path.read_text())
    return out


def main() -> None:
    agg = json.loads(AGG_PATH.read_text())
    comp = agg["compression"]["tcn"]["configs"]

    fp32_scores = comp["FP32"]["macro_f1"]["scores"]  # [fold][seed_idx]

    summary = {}
    for config in CONFIGS:
        real = load_real_tflite(config)
        n_found = len(real)
        proxy_scores = comp[config]["macro_f1"]["scores"]

        fp32_paired, proxy_paired, real_paired = [], [], []
        phase_f1_accum = {"LR": [], "LS": [], "PSw": [], "Sw": []}
        per_combo = []
        for seed_idx, seed in enumerate(SEEDS):
            for fold in FOLDS:
                key = (fold, seed)
                if key not in real:
                    continue
                r = real[key]
                fp32_paired.append(fp32_scores[fold][seed_idx])
                proxy_paired.append(proxy_scores[fold][seed_idx])
                real_paired.append(r["macro_f1"])
                for ph in phase_f1_accum:
                    if r.get("phase_f1", {}).get(ph) is not None:
                        phase_f1_accum[ph].append(r["phase_f1"][ph])
                per_combo.append({"fold": fold, "seed": seed, "macro_f1": r["macro_f1"]})

        real_arr = np.array(real_paired)
        fp32_arr = np.array(fp32_paired)
        proxy_arr = np.array(proxy_paired)
        gap_vs_fp32 = real_arr - fp32_arr
        gap_vs_proxy = real_arr - proxy_arr

        if len(real_arr) >= 2 and not np.allclose(real_arr, fp32_arr):
            stat, p = wilcoxon(real_arr, fp32_arr)
        else:
            p = None

        worst_idx = int(np.argmin(gap_vs_fp32)) if len(gap_vs_fp32) else None
        worst = per_combo[worst_idx] if worst_idx is not None else None

        summary[config] = {
            "n": n_found,
            "real_tflite_macro_f1_mean": float(real_arr.mean()) if n_found else None,
            "real_tflite_macro_f1_std": float(real_arr.std(ddof=1)) if n_found > 1 else None,
            "fp32_mean": float(fp32_arr.mean()) if n_found else None,
            "proxy_mean": float(proxy_arr.mean()) if n_found else None,
            "gap_vs_fp32_mean_pp": float(gap_vs_fp32.mean() * 100) if n_found else None,
            "gap_vs_fp32_std_pp": float(gap_vs_fp32.std(ddof=1) * 100) if n_found > 1 else None,
            "gap_vs_proxy_mean_pp": float(gap_vs_proxy.mean() * 100) if n_found else None,
            "wilcoxon_p_real_vs_fp32": float(p) if p is not None else None,
            "worst_case": worst,
            "worst_gap_pp": float(gap_vs_fp32.min() * 100) if n_found else None,
            "phase_f1_mean": {
                ph: (float(np.mean(v)) if v else None) for ph, v in phase_f1_accum.items()
            },
            "phase_f1_n": {ph: len(v) for ph, v in phase_f1_accum.items()},
        }

    out_path = RESEARCH_ROOT / "analysis" / "tflite_real_accuracy_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
