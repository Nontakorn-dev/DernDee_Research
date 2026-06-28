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

bash experiments/tinytcn/scripts/train.sh
bash analysis/scripts/plot_paper_figures.sh
bash experiments/compression/scripts/run_compression.sh
```

If `experiments/tinytcn/runs/fp32_100hz/best_model.pt` is missing, run the TinyTCN training command before compression.

## Status

- TinyTCN FP32: implemented; existing test report shows macro F1 about 0.91.
- Compression: runner added for INT8, INT4 simulation, Prune50, and INT8+Prune50; metrics require a local checkpoint and evaluation run.
- Baselines: folders exist, but models are stubs and should be treated as future work or external context.
- ESP32-S3: benchmark protocol is documented; latency/SRAM/power must be measured on hardware.

See `docs/PROTOCOL.md`, `EVALUATION.md`, and `PAPER_TABLES.md`.
