# Research Protocol

This document is the canonical workflow for the paper **Impact of Deep Compression Techniques on Phase-Specific Accuracy in TCN for Real-Time Gait Detection**.

## Research question

When TCN is compressed for microcontroller deployment, do transition phases (LR and PSw) lose more accuracy than steadier phases (LS and Sw)?

The first source of truth is PyTorch evaluation. ESP32-C3 latency and SRAM are reported only after hardware measurement.

## Fixed methodology

- Sampling rate: 100 Hz
- Source rate: 200 Hz, decimated by 2
- Window length: 50 samples (0.5 s)
- Training stride: 5 samples (0.05 s)
- Validation/test stride: 1 sample (highly overlapping test windows)
- Labels: LR, LS, PSw, Sw
- **Track B (primary):** subject-wise stratified 5-fold CV × 3 seeds `{42, 123, 456}`
- **Track A (exploratory):** single split seed 42 — appendix only
- Input: 12 bilateral shank IMU channels, no contact channels

## K-fold splits (Track B)

Generate stratified folds with **rotating validation per fold**:

```bash
python shared/scripts/make_kfold_splits.py --seed 42
python analysis/fold_qa_report.py
```

Outputs:

- `shared/splits/folds/fold{0..4}.csv`
- `shared/splits/folds/fold{K}_meta.json` (explicit train/val/test subject lists)
- `analysis/fold_qa_report.json` (phase balance + val overlap matrix)

Validation subjects are re-sampled from the non-test pool each fold (`fold_seed = base_seed + fold_index`).

## Split integrity assertions

Runtime checks in `shared/data/splits.py` enforce:

| Phase | Policy | Allowed data |
|-------|--------|--------------|
| Baseline FP32 train | `TRAIN_ONLY` | train split |
| Baseline val (checkpoint selection) | `VAL_ONLY` | val split |
| Prune fine-tune | `TRAIN_ONLY` | train split only (fixed epochs, no val) |
| Test eval | `TEST_ONLY` | test split |

Wired in: `train_runner.py`, `run_eval.py`, `eval_checkpoint.py`.

## Data preparation

Raw NONAN GaitPrint CSVs are converted to `dataset/Xy` with:

```bash
python shared/preprocess_gaitprint.py --input-dir NONAN_Dataset --output-dir dataset/Xy
python shared/scripts/upgrade_xy_labels.py --xy-root dataset/Xy
```

## FP32 model training (Track B)

Per fold×seed job:

```bash
python experiments/scripts/run_kfold_orchestrator.py --sync-only
caffeinate -dimsu bash experiments/scripts/run_overnight.sh --next --lazy
```

Run directory: `experiments/{model}/runs/fold{K}_seed{S}_fp32_100hz/`  
Split file: `shared/splits/folds/fold{K}.csv`

Canonical config template: `shared/configs/train_fair_comparison.json`

## Compression evaluation (Track B)

Uses FP32 checkpoint from matching fold×seed:

```bash
python experiments/compression/run_eval.py \
  --checkpoint experiments/tcn/runs/fold0_seed42_fp32_100hz/best_model.pt \
  --split-file shared/splits/folds/fold0.csv \
  --out-dir experiments/compression/runs/fold0_seed42
```

Configs (8 × 15 = 120 jobs): `FP32`, `INT8`, `Prune25`, `Prune50`, `Prune75`,
`INT8+Prune50`, `Prune50_ft15`, `Prune50_ft50` (default Prune50 fine-tune: 5 epochs).
INT4 is excluded (not deployable on ESP32-C3).

## Track A (exploratory, single split)

```bash
bash experiments/scripts/run_track_a_ablation.sh
bash experiments/scripts/run_track_a_prune_grid.sh
```

Not used for primary significance claims.

## Aggregation and statistics

```bash
python analysis/aggregate_runs.py
python analysis/subject_blocked_metrics.py --checkpoint ... --split-file ...
```

Reports: mean±std, σ_fold vs σ_seed, paired Wilcoxon/t-test.

## MacBook overnight guide

See [`docs/MACBOOK_TRAINING.md`](MACBOOK_TRAINING.md).

## ESP32-C3 measurement boundary

Do not fill ESP32 latency from PyTorch runs. Use `experiments/esp32/scripts/measure_deploy_candidates.py` to track pending hardware measurements.

## Paper table sources

- Table II / III: `analysis/aggregated_runs.json` (after Track B complete)
- Legacy single-split: `experiments/*/runs/fp32_100hz/test_metrics.json`
- ESP32 table: `experiments/esp32/benchmarks/esp32_c3_metrics.json`
