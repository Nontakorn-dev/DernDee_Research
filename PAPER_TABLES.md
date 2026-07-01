# Paper Tables Checklist

Living summary of measured results for `paper/main.tex`. All classification metrics are on the **held-out test split** unless noted. Percentages are rounded to two decimals for manuscript copy.

## Evaluation protocol (report in Methods)

| Item | Value | Source |
|------|-------|--------|
| Dataset | NONAN GaitPrint, bilateral shank IMU | `dataset/Xy/` |
| Input channels | 12 (acc + gyro, left/right shank) | `shared/gait_labels.py` |
| Sampling rate | 100 Hz (200 Hz decimated ×2) | `shared/configs/train_fair_comparison.json` |
| Window | 50 samples (0.5 s, causal) | same |
| Train / val / test stride | 5 / 1 / 1 samples | same |
| Subject split | 24 train / 5 val / 6 test (seed 42) | `shared/splits/subject_split.csv` |
| Selection metric | Best checkpoint by **validation macro F1** | `experiments/*/runs/fp32_100hz/history.json` |
| Test windows | 2,583,680 | all `test_metrics.json` |
| Phase support (test) | LR 151,735 · LS 1,274,499 · PSw 140,589 · Sw 1,016,857 | same |

---

## Table 1 — Baseline model comparison (FP32, fair protocol)

Primary source: `experiments/<model>/runs/fp32_100hz/test_metrics.json`

| Model | Params | FP32 size (KB) | Val macro F1 (%) | Test acc (%) | Test macro F1 (%) | Best epoch | Δ vs TinyTCN F1 (pp) |
|-------|--------|----------------|------------------|--------------|-------------------|------------|------------------------|
| **TinyTCN (ours)** | 10,948 | 42.77 | 91.03 | **96.30** | **91.50** | 48 | 0.00 |
| Transformer | 25,956 | 101.39 | 91.74 | 96.38 | 91.73 | 12 | +0.23 |
| LSTM+GRU | 12,356 | 48.27 | 91.36 | 96.25 | 91.38 | 29 | −0.12 |
| TCN | 11,560 | 45.16 | 91.67 | 96.04 | 90.96 | 12 | −0.54 |
| CNN1D | 10,727 | 41.90 | 91.05 | 95.58 | 90.23 | 42 | −1.27 |

Params for TinyTCN from `experiments/compression/manifest.json`; other models counted from `best_model.pt` (trainable weights, FP32 size = params × 4 B).

### Table 1b — Phase-specific test F1 (%) by architecture

| Phase | TinyTCN | Transformer | LSTM+GRU | TCN | CNN1D |
|-------|---------|-------------|----------|-----|-------|
| LR | **87.26** | 87.28 | **87.85** | 86.40 | 85.77 |
| LS | 97.88 | **98.07** | 97.91 | 97.71 | 97.54 |
| PSw | 82.99 | **83.84** | 82.00 | 82.01 | 80.29 |
| Sw | 97.87 | 97.72 | 97.78 | 97.73 | 97.34 |

Sources: `experiments/<model>/runs/fp32_100hz/test_metrics.json`

---

## Table 2 — TinyTCN FP32 test breakdown

Source: `experiments/tinytcn/runs/fp32_100hz/test_metrics.json`

| Phase | Support | F1 (%) | Phase accuracy (%) |
|-------|---------|--------|--------------------|
| LR | 151,735 | 87.26 | 95.03 |
| LS | 1,274,499 | 97.88 | 96.38 |
| PSw | 140,589 | 82.99 | 91.67 |
| Sw | 1,016,857 | 97.87 | 97.03 |
| **Overall** | 2,583,680 | **91.50** (macro F1) | **96.30** (acc) |

---

## Table 3 — Compression vs phase F1 (PyTorch test set)

Run (already completed):

```bash
bash experiments/compression/scripts/run_compression.sh
```

Primary sources:

- `experiments/compression/manifest.json`
- `experiments/compression/results/metrics.json`

Estimated payload size (KB) uses the compression pipeline formula; **TFLite file size** is measured on export (`experiments/esp32/exports/`).

| Config | Params | Est. size (KB) | TFLite (KB) | Test acc (%) | Macro F1 (%) | Δ F1 vs FP32 (pp) | LR F1 | LS F1 | PSw F1 | Sw F1 | Method note |
|--------|--------|----------------|-------------|--------------|--------------|-------------------|-------|-------|-------|-------|-------------|
| FP32 | 10,948 | 42.77 | 49.21 | 96.30 | 91.50 | 0.00 | 87.26 | 97.88 | 82.99 | 97.87 | FP32 checkpoint |
| INT8 | 10,948 | 12.69 | 18.95 | 96.27 | 91.43 | −0.07 | 87.04 | 97.84 | 82.95 | 97.87 | PyTorch INT8 QDQ proxy |
| Prune50 | 3,428 | 13.39 | 20.29 | 94.74 | 88.73 | −2.77 | 81.32 | 96.88 | 79.67 | 97.03 | 50% hidden prune + fine-tune |
| INT8+Prune50 | 3,428 | 5.35 | 11.57 | 94.56 | 88.45 | −3.05 | 80.52 | 96.76 | 79.56 | 96.95 | Prune50 + INT8 QDQ |

---

## Table 4 — Phase-specific accuracy under compression

Source: `experiments/compression/phase_degradation.json` (metric: phase accuracy, Δ relative to FP32)

| Config | Test acc (%) | Macro F1 (%) | LR acc (%) | Δ LR (pp) | LS acc (%) | Δ LS (pp) | PSw acc (%) | Δ PSw (pp) | Sw acc (%) | Δ Sw (pp) |
|--------|--------------|--------------|------------|-----------|------------|-----------|-------------|------------|------------|-----------|
| FP32 | 96.30 | 91.50 | 95.03 | 0.00 | 96.38 | 0.00 | 91.67 | 0.00 | 97.03 | 0.00 |
| INT8 | 96.27 | 91.43 | 95.20 | +0.17 | 96.28 | −0.10 | 91.18 | −0.50 | 97.11 | +0.08 |
| Prune50 | 94.74 | 88.73 | 97.39 | +2.36 | 94.36 | −2.02 | 94.11 | +2.44 | 94.90 | −2.13 |
| INT8+Prune50 | 94.56 | 88.45 | 97.62 | +2.59 | 94.10 | −2.28 | 94.41 | +2.74 | 94.71 | −2.32 |

**Interpretation for Discussion:** Aggressive prune (especially INT8+Prune50) hurts **PSw** F1 most; LR/PSw are transition phases. Phase *accuracy* can rise while *F1* falls when precision drops — report both in the paper.

---

## Table 5 — On-device deployment (ESP32-C3)

Source: `experiments/esp32/benchmarks/esp32_c3_metrics.json`  
Measured: 2026-07-01 · Board: ESP32-C3 Dev Module @ 160 MHz · Toolchain: Arduino IDE · Library: Chirale_TensorFlowLite  
Protocol: synthetic replay window (50×12), 20 warmup + 1000 timed `Invoke()` trials, tensor arena 120 KB.

| Config | TFLite (KB) | Latency mean (ms) | Latency p95 (ms) | Min free heap (KB) | Replay pred | Label | Match | Host TFLite smoke |
|--------|-------------|-------------------|------------------|--------------------|-------------|-------|-------|-------------------|
| INT8+Prune50 | 11.57 | **69.4** | 69.4 | 156.3 | Sw | Sw | ✓ | ✓ (`host_validation.json`) |
| INT8 | 18.95 | 256.8 | 256.8 | 156.3 | Sw | Sw | ✓ | ✓ |
| Prune50 | 20.29 | 260.0 | 260.0 | 156.3 | Sw | Sw | ✓ | ✓ |
| FP32 | 49.21 | 850.6 | 850.6 | 156.3 | Sw | Sw | ✓ | ✓ |

---

## Table 6 — Host PyTorch vs TFLite smoke validation

Source: `experiments/esp32/benchmarks/host_validation.json`  
Fixed replay window index 0 (label Sw, trial `S002_G01_D01_B01_T01.csv`).

| Config | PyTorch pred | TFLite pred | Argmax match | Max \|Δlogit\| | Tolerance | Pass |
|--------|--------------|-------------|--------------|----------------|-----------|------|
| FP32 | Sw | Sw | ✓ | 1.9×10⁻⁶ | 0.05 | ✓ |
| INT8 | Sw | Sw | ✓ | 0.626 | 1.00 | ✓ |
| Prune50 | Sw | Sw | ✓ | 1.9×10⁻⁶ | 0.05 | ✓ |
| INT8+Prune50 | Sw | Sw | ✓ | 0.382 | 1.00 | ✓ |

---

## Figure checklist

```bash
bash analysis/scripts/plot_paper_figures.sh
```

Outputs in `paper/figures/`:

| Figure | File | Status | Source |
|--------|------|--------|--------|
| Pareto front | `pareto_front.pdf` | Ready | `experiments/compression/manifest.json` |
| Phase degradation | `phase_degradation.pdf` | Ready | `experiments/compression/phase_degradation.json` |
| Confusion matrix | `confusion_matrix.pdf` | Ready | `experiments/tinytcn/runs/fp32_100hz/test_confusion_matrix.json` + compression runs |
| Training convergence | `training_convergence.pdf` | Ready | `experiments/tinytcn/runs/fp32_100hz/history.json` |

Individual commands:

```bash
python analysis/plot_pareto.py
python analysis/plot_phase_degradation.py --metric phase_accuracy
python analysis/plot_confusion_matrix.py
python analysis/plot_training.py
```

---

## Manuscript notes (IEEE credibility)

1. **Report both macro F1 and phase-specific F1** — gait phases are imbalanced (LS/Sw dominate support).
2. **Distinguish PyTorch compression metrics from TFLite deployment size** — Table 3 includes both estimated payload and measured `.tflite` bytes.
3. **Hardware table must state board, CPU freq, trials, and arena size** — already captured in Table 5 header.
4. **Real-time budget** — define per deployment stride (e.g. 50 ms for stride-5 @ 100 Hz); INT8+Prune50 mean latency is 69.4 ms on ESP32-C3.
5. **Pending placeholders for paper only:** optional power (mW) from current draw.

Deployable MCU configs in scope: **FP32, INT8, Prune50, INT8+Prune50** (see `experiments/esp32/README.md`).

### Suggested LaTeX table captions mapping

| This file | `paper/main.tex` label |
|-----------|-------------------------|
| Table 1 / 1b | `\label{tab:baseline}` |
| Table 3 + 4 | `\label{tab:phase-degradation}` |
| Table 5 | `\label{tab:esp32}` |
