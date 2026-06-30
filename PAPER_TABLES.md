# Paper Tables Checklist

## Table 1 — Model complexity

| Model | Params | Size (KB) | Status |
|-------|--------|-----------|--------|
| TinyTCN FP32 | 10,948 | 42.8 | `experiments/tinytcn/runs/fp32_100hz/test_report.txt`, `analysis/model_profile.py` |
| INT8 / INT4 / Prune50 / INT8+Prune50 | measured or estimated | manifest | `experiments/compression/manifest.json` |

## Table 2 — FP32 classification (12 ch, 100 Hz)

Source: `experiments/<model>/runs/fp32_100hz/test_report.txt` after retraining with shared config.

| Phase | TinyTCN F1 |
|-------|------------|
| LR | pending |
| LS | pending |
| PSw | pending |
| Sw | pending |

## Table 3 — Compression vs phase F1

Run:

```bash
bash experiments/compression/scripts/run_compression.sh
python analysis/phase_degradation.py --metric phase_accuracy
```

Primary sources:

- `experiments/compression/results/metrics.json`
- `experiments/compression/manifest.json`
- `experiments/compression/phase_degradation.json`

| Config | Size KB | Macro F1 | LR | LS | PSw | Sw |
|--------|---------|----------|----|----|-----|-----|
| FP32 | 42.8 | pending | pending | pending | pending | pending |
| INT8 | 12.7 estimate until run | pending | pending | pending | pending | pending |
| INT4 | 7.3 estimate until run | pending | pending | pending | pending | pending |
| Prune50 | 21.4 estimate until run | pending | pending | pending | pending | pending |
| INT8+Prune50 | 6.3 estimate until run | pending | pending | pending | pending | pending |

## Table 4 — ESP32-S3 deployment

| Config | Latency (ms) | SRAM (KB) | Status |
|--------|--------------|-----------|--------|
| INT8 | pending hardware measurement | pending hardware measurement | TODO |
| INT8+Prune50 | pending hardware measurement | pending hardware measurement | TODO |

## Figure — Pareto front

```bash
bash analysis/scripts/plot_paper_figures.sh
```

Outputs in `paper/figures/`:

| Figure | File | When available |
|--------|------|----------------|
| Pareto front | `pareto_front.pdf` | after FP32; full curve after compression |
| Phase degradation | `phase_degradation.pdf` | after FP32; full bars after compression |
| Confusion matrix | `confusion_matrix.pdf` | after FP32 train (val + test); adds compressed configs after compression |
| Training convergence | `training_convergence.pdf` | after FP32 train |

Individual commands:

```bash
python analysis/plot_pareto.py
python analysis/plot_phase_degradation.py --metric phase_accuracy
python analysis/plot_confusion_matrix.py
python analysis/plot_training.py
```
