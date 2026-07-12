# Compression experiments

PyTorch-first INT8 quantization proxy and structured pruning experiments for TinyTCN.

The purpose of this folder is to measure **phase-specific accuracy degradation** before any ESP32 deployment work. INT8 here is a PyTorch quantize-dequantize accuracy proxy unless a later export step explicitly says otherwise.

## Status

| Step | Script | Status |
|------|--------|--------|
| Pareto manifest + plot | `scripts/plot_pareto.sh` | Ready |
| Per-config PyTorch eval | `run_eval.py` | Ready |
| INT8 accuracy proxy | `run_eval.py --configs INT8` | Ready |
| Structured Prune50 | `run_eval.py --configs Prune50` | Ready |
| INT8+Prune50 | `run_eval.py --configs INT8+Prune50` | Ready |
| TFLite/ESP32 export | `../esp32/README.md` | Ready |

## Run compression evaluation

### Primary: k-fold × seed (15 checkpoints, paper protocol)

```bash
# Full pipeline: clean legacy runs → 120 compression jobs → aggregate
mkdir -p logs && caffeinate -dimsu bash experiments/scripts/run_kfold_compression.sh --lazy \
  > logs/kfold_compression.log 2>&1

# Monitor
tail -f logs/kfold_compression.log
```

Outputs per fold×seed:

```
experiments/compression/runs/fold{0..4}_seed{42,123,456}/
├── FP32/metrics.json
├── INT8/metrics.json
├── Prune50/metrics.json
└── INT8+Prune50/metrics.json
```

Aggregate mean ± std (N=15):

```bash
python analysis/aggregate_runs.py
```

Requires `best_model.pt` in each `experiments/tinytcn/runs/fold*_seed*_fp32_100hz/`.
The script retrains missing baselines automatically unless `--skip-baseline` is set.

### Legacy single-split (exploratory only)

```bash
bash experiments/compression/scripts/run_compression.sh
```

This requires:

- `experiments/tinytcn/runs/fp32_100hz/best_model.pt`
- `experiments/tinytcn/runs/fp32_100hz/norm_stats.json` or checkpoint-embedded norm stats
- `shared/splits/subject_split.csv`
- `dataset/Xy`

K-fold configs (see `kfold_configs.py`): **FP32**, **INT8**, **Prune25/50/75**, **INT8+Prune50**, plus fine-tune ablation **Prune50\_ft15/ft50**. INT4 is excluded (not deployable on ESP32-C3).

Models: **tinytcn** (default) and **tcn** via `--model tcn`.

### TCN-family ablation sweep (both models)

```bash
bash experiments/scripts/run_tcn_family_ablation.sh
```

Runs on the legacy split:
- Pareto grid: FP32, INT8, INT4, prune ratios 25/50/75%, combined INT8/INT4+Prune
- Fine-tune ablation: Prune50 and INT8+Prune50 with 5/15/50 epochs

Outputs → `experiments/compression/runs/tcn_family_ablation/`  
Summary → `analysis/ablation_summary.json`, `analysis/ABLATION_TABLES.md`, `paper/tables/ablation_*.tex`

Outputs:

```
experiments/compression/
├── runs/
│   ├── FP32/metrics.json
│   ├── INT8/metrics.json
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
