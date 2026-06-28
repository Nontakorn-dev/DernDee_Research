# Experiments

One folder per model architecture.

```
experiments/
├── tinytcn/        ✅ model, train, inference, runs/
├── cnn1d/          📋 TODO
├── lstm_gru/       📋 TODO
├── tcn/            📋 TODO
├── transformer/    📋 TODO
├── compression/    Pareto + quantize/prune
└── esp32/          On-device deploy notes
```

## Train (subject split)

```bash
bash experiments/tinytcn/scripts/train.sh
```

All models use `shared/splits/subject_split.csv` for comparable train/val/test.

## Pareto

```bash
bash experiments/compression/scripts/plot_pareto.sh
```
