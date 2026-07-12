# Paper Tables Checklist

Living summary of measured results for `paper/main.tex`. Regenerate compression TeX after aggregation:

```bash
python analysis/aggregate_runs.py          # also refreshes paper/tables/kfold_compression.tex
python analysis/export_kfold_compression_tex.py
```

## Status summary (2026-07-12)

Numbering matches **compiled PDF order** in `paper/main.pdf`:

| PDF Table | Label | Source | Status |
|-----------|-------|--------|--------|
| **I** | `tab:compression` | `main.tex` | **Complete** |
| **II** | `tab:baseline` | `aggregated_runs.json` | **Complete** ($N{=}15$) |
| **III** | `tab:kfold-compression` | `paper/tables/kfold_compression.tex` | **Complete** (120/120) |
| **IV** | `tab:pareto-grid` | `paper/tables/ablation_pareto.tex` | **Complete** |
| **V** | `tab:finetune-ablation` | `paper/tables/ablation_finetune.tex` | **Complete** |
| **VI** | `tab:esp32` | `paper/tables/esp32_placeholder.tex` | **Complete** (4/4 configs) |

---

## Table I — Baseline FP32 ($K{=}5 \times 3$, primary)

| Model | Params | FP32 KB | Macro F1 (%) | σ overall |
|-------|--------|---------|--------------|-----------|
| **TCN (primary)** | 11,560 | 45.2 | **90.87 ± 1.47** | 1.47 |
| Transformer | 25,956 | 101.4 | 90.83 ± 0.99 | 0.99 |
| LSTM+GRU | 12,356 | 48.3 | 90.59 ± 1.19 | 1.19 |
| CNN1D | 10,727 | 41.9 | 90.34 ± 1.21 | 1.21 |

Wilcoxon (TCN vs.): Transformer $p{=}0.80$; LSTM+GRU $p{=}0.30$; CNN1D $p{=}0.073$.

---

## Table II — TCN k-fold compression (primary, placeholder until $N{=}15$)

Auto-generated in `paper/tables/kfold_compression.tex`. Protocol: **8 configs × 15 runs = 120 jobs**.

Primary Pareto (6): FP32, INT8, Prune25, Prune50, Prune75, INT8+Prune50  
Fine-tune ablation (2): Prune50\_ft15, Prune50\_ft50 (Prune50 = 5 epochs)

**Total:** 120/120 jobs complete.

```bash
bash experiments/scripts/run_kfold_compression.sh --model tcn --skip-clean --lazy --reset-running
```

---

## Table III — Exploratory preview (fold0, seed42 only)

Source: `experiments/compression/runs/tcn/fold0_seed42/*/metrics.json`

| Config | Macro F1 (%) | Δ vs FP32 | PSw F1 (%) |
|--------|--------------|-----------|------------|
| FP32 | 91.07 | — | 82.80 |
| INT8 | 91.10 | +0.03 | 82.91 |
| Prune50 | 90.67 | −0.41 | 81.95 |
| INT8+Prune50 | 89.74 | −1.34 | 80.77 |

Not used for primary Wilcoxon claims.

---

## Table IV — Phase accuracy preview (fold0, seed42)

| Config | LR | LS | PSw | Sw |
|--------|-----|-----|-----|-----|
| FP32 | 97.83 | 96.36 | 95.81 | 95.14 |
| INT8 | 97.84 | 96.37 | 95.77 | 95.16 |
| Prune50 | 97.70 | 96.85 | 94.02 | 94.35 |
| INT8+Prune50 | 97.92 | 96.29 | 93.30 | 93.87 |

---

## Table VI — ESP32-C3 on-device inference

Source: `experiments/esp32/benchmarks/esp32_c3_metrics.json` (measured 2026-07-12).

Checkpoint: `fold0_seed42`. Arena: 120 KB (INT8, Prune50, INT8+Prune50) / 160 KB (FP32).

| Config | Mean (ms) | p95 (ms) | TFLite (KB) | Min heap (KB) |
|--------|-----------|----------|-------------|---------------|
| FP32 | 856.2 | 856.2 | 52.1 | 116.3 |
| INT8 | 279.8 | 279.8 | 21.3 | 156.3 |
| Prune50 | 253.0 | 253.0 | 21.6 | 156.3 |
| **INT8+Prune50** | **75.7** | **75.7** | **13.4** | **156.3** |

**Deploy recommendation:** INT8+Prune50 (fastest, smallest, real-time within 500 ms window).

---

## Figures

| Figure | Status | Notes |
|--------|--------|-------|
| `pareto_front.pdf` | **Current** | Regenerated from `manifest_fold0_seed42.json` |
| `phase_degradation.pdf` | **Current** | Same source (fold~0, seed~42) |
| Architecture (TikZ) | **Current** | Standard TCN 4-block diagram in `main.tex` |
