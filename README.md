# TinyTCN Compression for Real-Time Gait Detection

Research repo for the paper **Impact of Deep Compression Techniques on Phase-Specific Accuracy in TinyTCN for Real-Time Gait Detection**.

The project studies how TinyTCN compression affects each gait phase (LR, LS, PSw, Sw) when moving toward ESP32-S3 deployment. The current research contract is:

- 12-channel bilateral shank IMU input
- 100 Hz evaluation only
- 0.5 s causal windows
- subject-wise split: 24 train / 5 validation / 6 test
- PyTorch-first compression evaluation for accuracy and phase degradation
- ESP32-S3 measurements only after hardware benchmarking

## Layout

```
Research/
├── paper/                  # LaTeX draft and generated figures
├── dataset/Xy              # Symlink to preprocessed trial CSVs
├── docs/                   # Reproducibility protocol
├── shared/                 # Labels, data loading, splits, metrics
├── analysis/               # Pareto and phase-degradation artifacts
└── experiments/
    ├── tinytcn/            # Implemented FP32 model, train, inference
    ├── compression/        # PyTorch-first compression evaluation
    ├── esp32/              # Hardware benchmark protocol
    └── */                  # Baseline stubs / future work
```

## Quickstart

Run commands from this repository root:

```bash
pip install -r shared/requirements.txt

bash experiments/scripts/train_all_colab.sh \
  --config shared/configs/train_fair_comparison.json \
  --data-root dataset/Xy \
  --lazy \
  --device cuda
```

If `experiments/tinytcn/runs/fp32_100hz/best_model.pt` is missing, train FP32 models first (see `docs/COLAB_TRAINING.md`), then run compression and figure scripts.

## Status

- TinyTCN + baselines: shared training runner implemented; retrain all models with `experiments/scripts/train_all_colab.sh`.
- Compression: runner for INT8, INT4, Prune50, INT8+Prune50; metrics pending fresh FP32 checkpoint.
- ESP32-S3: benchmark protocol documented; latency/SRAM/power require hardware measurement.

See `docs/PROTOCOL.md`, `EVALUATION.md`, and `PAPER_TABLES.md`.
