# Compression experiments

PyTorch-first INT8, 4-bit simulation, and structured pruning experiments for TinyTCN.

The purpose of this folder is to measure **phase-specific accuracy degradation** before any ESP32 deployment work. INT8 and INT4 here are PyTorch quantize-dequantize accuracy proxies unless a later export step explicitly says otherwise.

## Status

| Step | Script | Status |
|------|--------|--------|
| Pareto manifest + plot | `scripts/plot_pareto.sh` | Ready |
| Per-config PyTorch eval | `run_eval.py` | Ready |
| INT8 accuracy proxy | `run_eval.py --configs INT8` | Ready |
| INT4 accuracy floor | `run_eval.py --configs INT4` | Ready |
| Structured Prune50 | `run_eval.py --configs Prune50` | Ready |
| INT8+Prune50 | `run_eval.py --configs INT8+Prune50` | Ready |
| TFLite/ESP32 export | `../esp32/README.md` | TODO |

## Run compression evaluation

```bash
bash experiments/compression/scripts/run_compression.sh
```

This requires:

- `experiments/tinytcn/runs/fp32_100hz/best_model.pt`
- `experiments/tinytcn/runs/fp32_100hz/norm_stats.json` or checkpoint-embedded norm stats
- `shared/splits/subject_split.csv`
- `dataset/Xy`

Outputs:

```
experiments/compression/
├── runs/
│   ├── FP32/metrics.json
│   ├── INT8/metrics.json
│   ├── INT4/metrics.json
│   ├── Prune50/metrics.json
│   └── INT8+Prune50/metrics.json
├── results/metrics.json
├── overrides.json
├── manifest.json
└── phase_degradation.json
```

## Pareto and phase degradation

```bash
python analysis/collect_pareto.py
python analysis/plot_pareto.py
python analysis/phase_degradation.py --metric phase_accuracy
```

Generated figures:

- `paper/figures/pareto_front.pdf`
- `paper/figures/pareto_front.png`

## Metrics schema

Each `metrics.json` includes:

- `accuracy`
- `macro_f1`
- `phase_f1`
- `phase_accuracy`
- `phase_support`
- `size_kb`
- `params`
- `artifact`
- `method_note`

Reads FP32 metrics from `experiments/tinytcn/runs/fp32_100hz/`.
