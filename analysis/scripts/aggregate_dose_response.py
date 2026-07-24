#!/usr/bin/env python3
"""Aggregate real-TFLite vs. clean-PyTorch-proxy stats for all combined
INT8+Prune* configs (dose-response grid), N=15 each.

INT8+Prune25/75 proxy comes from analysis/clean_prune_int8_proxy_results
(QDQ applied directly to the existing Prune25/75 checkpoint, no re-finetune).
INT8+Prune50 proxy comes from analysis/aggregated_runs.json (the original,
independently re-finetuned checkpoint) for continuity with Tables II/III
already in the paper -- see Limitations for the resulting ~0.3pp caveat.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
REAL_DIR = RESEARCH_ROOT / "analysis" / "tflite_real_accuracy_results"
CLEAN_PROXY_DIR = RESEARCH_ROOT / "analysis" / "clean_prune_int8_proxy_results"
AGG_PATH = RESEARCH_ROOT / "analysis" / "aggregated_runs.json"

SEEDS = [42, 123, 456]
FOLDS = [0, 1, 2, 3, 4]
CONFIGS = ["INT8", "INT8+Prune75", "INT8+Prune50", "INT8+Prune25"]


def load_json_dir(dir_: Path, config: str) -> dict[tuple[int, int], dict]:
    tag = config.replace("+", "_plus_")
    out = {}
    for seed in SEEDS:
        for fold in FOLDS:
            path = dir_ / f"{tag}_fold{fold}_seed{seed}.json"
            if path.exists():
                out[(fold, seed)] = json.loads(path.read_text())
    return out


def main() -> None:
    agg = json.loads(AGG_PATH.read_text())
    comp = agg["compression"]["tcn"]["configs"]
    fp32_scores = comp["FP32"]["macro_f1"]["scores"]  # [fold][seed_idx]

    summary = {}
    for config in CONFIGS:
        real = load_json_dir(REAL_DIR, config)
        if config == "INT8+Prune50":
            proxy_scores_grid = comp["INT8+Prune50"]["macro_f1"]["scores"]
            proxy_lookup = {
                (fold, SEEDS[si]): proxy_scores_grid[fold][si]
                for fold in FOLDS
                for si in range(3)
            }
        elif config == "INT8":
            proxy_scores_grid = comp["INT8"]["macro_f1"]["scores"]
            proxy_lookup = {
                (fold, SEEDS[si]): proxy_scores_grid[fold][si]
                for fold in FOLDS
                for si in range(3)
            }
        else:
            clean = load_json_dir(CLEAN_PROXY_DIR, config)
            proxy_lookup = {k: v["macro_f1"] for k, v in clean.items()}

        fp32_paired, proxy_paired, real_paired = [], [], []
        per_combo = []
        for seed_idx, seed in enumerate(SEEDS):
            for fold in FOLDS:
                key = (fold, seed)
                if key not in real or key not in proxy_lookup:
                    continue
                fp32_paired.append(fp32_scores[fold][seed_idx])
                proxy_paired.append(proxy_lookup[key])
                real_paired.append(real[key]["macro_f1"])
                per_combo.append({"fold": fold, "seed": seed, "macro_f1": real[key]["macro_f1"]})

        n = len(real_paired)
        real_arr = np.array(real_paired)
        fp32_arr = np.array(fp32_paired)
        proxy_arr = np.array(proxy_paired)
        gap_vs_fp32 = real_arr - fp32_arr
        gap_vs_proxy = real_arr - proxy_arr

        if n >= 2 and not np.allclose(real_arr, fp32_arr):
            _, p = wilcoxon(real_arr, fp32_arr)
        else:
            p = None

        worst_idx = int(np.argmin(gap_vs_fp32)) if n else None
        worst = per_combo[worst_idx] if worst_idx is not None else None

        summary[config] = {
            "n": n,
            "real_mean": float(real_arr.mean() * 100) if n else None,
            "real_std": float(real_arr.std(ddof=1) * 100) if n > 1 else None,
            "gap_vs_fp32_mean_pp": float(gap_vs_fp32.mean() * 100) if n else None,
            "gap_vs_fp32_std_pp": float(gap_vs_fp32.std(ddof=1) * 100) if n > 1 else None,
            "gap_vs_proxy_mean_pp": float(gap_vs_proxy.mean() * 100) if n else None,
            "gap_vs_proxy_std_pp": float(gap_vs_proxy.std(ddof=1) * 100) if n > 1 else None,
            "wilcoxon_p": float(p) if p is not None else None,
            "worst_case": worst,
            "worst_gap_pp": float(gap_vs_fp32.min() * 100) if n else None,
        }

    print(json.dumps(summary, indent=2))
    out_path = RESEARCH_ROOT / "analysis" / "dose_response_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
