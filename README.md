# TCN Compression for Real-Time Gait Detection

Research repo for the paper **Impact of Deep Compression Techniques on Phase-Specific Accuracy in TCN for Real-Time Gait Detection**.

The project studies how TCN compression affects each gait phase (LR, LS, PSw, Sw) when moving toward ESP32-C3 deployment. The current research contract is:

- 100 Hz bilateral shank IMU (12 channels)
- 0.5 s causal windows (50 samples)
- Subject-wise stratified 5-fold CV × 3 seeds (primary)
- FP32 baselines: TCN, CNN1D, LSTM+GRU, Transformer, CNN-LSTM (hidden width 32)
- Compression: FP32, INT8, Prune50, INT8+Prune50

## Layout

```
experiments/
    ├── tcn/              # Primary deployment model
    ├── cnn1d/            # Baseline
    ├── cnn_lstm/         # Conv1d → LSTM hybrid baseline
    ├── lstm_gru/         # Baseline
    ├── transformer/      # Baseline
    ├── compression/      # PyTorch compression eval
    └── esp32/            # TFLite export + on-device benchmark
shared/                   # Training protocol, splits, eval
analysis/                 # Aggregation + paper figures
paper/                    # LaTeX manuscript
```

## Quick start

```bash
# Sync k-fold manifest
python experiments/scripts/run_kfold_orchestrator.py --sync-only

# TCN k-fold compression (120 jobs)
bash experiments/scripts/run_kfold_compression.sh --model tcn --lazy --skip-clean

# Aggregate results
python analysis/aggregate_runs.py
```

See `docs/PROTOCOL.md` for the full workflow.
