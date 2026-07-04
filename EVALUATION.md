# Evaluation Protocol

This repo evaluates TinyTCN compression at **100 Hz** with **0.5 s causal windows**.

## Track B — primary (Table II + III)

- **5-fold stratified subject-wise CV** with rotating validation per fold
- **3 seeds:** `{42, 123, 456}`
- Same protocol for baseline FP32 models and compression configs
- Report **mean ± std** across folds×seeds
- **Paired Wilcoxon / paired t-test** (paired by fold×seed); do not use CI-overlap heuristics
- **Variance decomposition:** σ_fold (subject heterogeneity) vs σ_seed (init noise) via `analysis/aggregate_runs.py`

Generate splits:

```bash
python shared/scripts/make_kfold_splits.py
python analysis/fold_qa_report.py
```

Run jobs:

```bash
bash experiments/scripts/run_overnight.sh --next --lazy
```

## Track A — exploratory (appendix)

- Single split `shared/splits/subject_split.csv` (seed 42)
- Fine-tune ablation: epochs `{5, 15, 50}` × lr `{1e-4, 5e-4}`
- Prune ratio grid: `{25%, 50%, 75%}`
- No significance claims

## Window overlap disclosure

Test/val stride = 1 on 50-sample windows → ~98% overlap between consecutive windows. Report:

- Window-level metrics (legacy tables)
- **Per-subject blocked macro F1** (`analysis/subject_blocked_metrics.py`)
- Bootstrap CI blocked by subject (not by windows)

## Split integrity

| Phase | Data | Val usage |
|-------|------|-----------|
| Baseline train | train | val for checkpoint selection |
| Prune fine-tune | train only | not used |
| Test eval | test | — |

Runtime assertions: `SplitPolicy` in `shared/data/splits.py`.

## Metrics

- overall accuracy
- macro F1
- phase-specific F1 for LR, LS, PSw, Sw
- phase-specific accuracy (diagonal / support)
- model size in KB
- on-device latency mean + p95 (ESP32 only)

## Compression configs

```bash
bash experiments/compression/scripts/run_compression.sh
```

Or per fold×seed:

```bash
python experiments/compression/run_eval.py \
  --checkpoint experiments/tinytcn/runs/fold0_seed42_fp32_100hz/best_model.pt \
  --split-file shared/splits/folds/fold0.csv \
  --out-dir experiments/compression/runs/fold0_seed42 \
  --keep-ratio 0.5
```

## Figures

```bash
bash analysis/scripts/plot_paper_figures.sh
```

Track B Pareto points should include error bars from k-fold aggregation when available.

## Statistical rules

1. Compare configs with paired tests on fold×seed pairs (n=15 for K=5, 3 seeds)
2. Report σ_fold and σ_seed in Discussion when interpreting variance
3. Flag low PSw support folds from `analysis/fold_qa_report.json`
