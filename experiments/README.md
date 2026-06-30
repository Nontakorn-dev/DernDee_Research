# Experiments

One folder per model architecture.

```
experiments/
├── tinytcn/        ✅ model, train, inference, runs/
├── cnn1d/          ✅ model, train
├── lstm_gru/       ✅ model, train
├── tcn/            ✅ model, train
├── transformer/    ✅ model, train
├── compression/    Pareto + quantize/prune
├── scripts/        train_all_colab.sh
└── esp32/          On-device deploy notes
```

## Train (shared fair-comparison protocol)

All models use `shared/configs/train_fair_comparison.json` and `shared/splits/subject_split.csv`.

Single model:

```bash
bash experiments/tinytcn/scripts/train.sh \
  --config shared/configs/train_fair_comparison.json
```

All models (Colab-friendly):

```bash
bash experiments/scripts/train_all_colab.sh \
  --config shared/configs/train_fair_comparison.json \
  --data-root dataset/Xy \
  --lazy \
  --device cuda
```

See `docs/COLAB_TRAINING.md` for Google Colab setup.

## Pareto

```bash
bash experiments/compression/scripts/plot_pareto.sh
```
