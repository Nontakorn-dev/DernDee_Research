# Research Protocol

This document is the canonical workflow for the paper **Impact of Deep Compression Techniques on Phase-Specific Accuracy in TinyTCN for Real-Time Gait Detection**.

## Research question

When TinyTCN is compressed for microcontroller deployment, do transition phases (LR and PSw) lose more accuracy than steadier phases (LS and Sw)?

The first source of truth is PyTorch evaluation. ESP32-C3 latency and SRAM are reported only after hardware measurement.

## Fixed methodology

- Sampling rate: 100 Hz
- Source rate: 200 Hz, decimated by 2
- Window length: 50 samples (0.5 s)
- Training stride: 5 samples (0.05 s)
- Validation/test stride: 1 sample
- Labels: LR, LS, PSw, Sw
- Split: subject-wise 24 train / 5 validation / 6 test
- Input: 12 bilateral shank IMU channels, no contact channels

## Data preparation

Raw NONAN GaitPrint CSVs are converted to `dataset/Xy` with:

```bash
python shared/preprocess_gaitprint.py --input-dir NONAN_Dataset --output-dir dataset/Xy
python shared/scripts/upgrade_xy_labels.py --xy-root dataset/Xy
```

Expected columns:

- `time`
- left shank accelerometer: `lt_acc_x`, `lt_acc_y`, `lt_acc_z`
- left shank gyroscope: `lt_gyr_x`, `lt_gyr_y`, `lt_gyr_z`
- right shank accelerometer: `rt_acc_x`, `rt_acc_y`, `rt_acc_z`
- right shank gyroscope: `rt_gyr_x`, `rt_gyr_y`, `rt_gyr_z`
- labels: `phase_lt`, `phase_rt`

Check labels:

```bash
python shared/scripts/check_xy_labels.py --channels bilateral
python shared/scripts/check_xy_labels.py --channels right
```

## Subject split

The split lives at `shared/splits/subject_split.csv`. Regenerate only when intentionally changing the protocol:

```bash
python shared/scripts/make_split.py --seed 42
```

## FP32 model training (shared protocol)

Canonical config: `shared/configs/train_fair_comparison.json`

Train one model:

```bash
bash experiments/tinytcn/scripts/train.sh \
  --config shared/configs/train_fair_comparison.json
```

Train all baselines:

```bash
bash experiments/scripts/train_all_colab.sh \
  --config shared/configs/train_fair_comparison.json \
  --lazy \
  --device cuda
```

Colab guide: `docs/COLAB_TRAINING.md`

Training defaults: 50 epochs, Adam lr=1e-3, batch size 512, balanced class weights. The checkpoint with the best validation macro F1 is saved as `best_model.pt` (not necessarily the final epoch). Target label for bilateral input is left-foot phase (`phase_lt`).

Primary outputs:

- `experiments/tinytcn/runs/fp32_100hz/best_model.pt`
- `experiments/tinytcn/runs/fp32_100hz/norm_stats.json`
- `experiments/tinytcn/runs/fp32_100hz/test_report.txt`
- `experiments/tinytcn/runs/fp32_100hz/test_confusion_matrix.json`
- `experiments/tinytcn/runs/fp32_100hz/history.json`

Evaluate:

```bash
python shared/eval_checkpoint.py \
  --checkpoint experiments/tinytcn/runs/fp32_100hz/best_model.pt \
  --out experiments/tinytcn/runs/fp32_100hz/test_metrics.json
```

## Compression evaluation

Run all configured compression experiments:

```bash
bash experiments/compression/scripts/run_compression.sh
```

Configs:

- `FP32`: baseline checkpoint evaluation
- `INT8`: PyTorch weight quantize-dequantize accuracy proxy
- `Prune50`: structured 50% hidden-channel pruning with fine-tuning
- `INT8+Prune50`: Prune50 with fine-tuning, then INT8 proxy

Outputs:

- `experiments/compression/runs/<CONFIG>/metrics.json`
- `experiments/compression/results/metrics.json`
- `experiments/compression/overrides.json`

## Metrics

Report these for each config:

- overall accuracy
- macro F1
- per-phase F1
- per-phase accuracy, computed as confusion-matrix diagonal divided by support
- model size in KB
- parameter count

Generate paper artifacts:

```bash
python analysis/collect_pareto.py
python analysis/plot_pareto.py
python analysis/phase_degradation.py --metric phase_accuracy
```

Outputs:

- `experiments/compression/manifest.json`
- `experiments/compression/phase_degradation.json`
- `paper/figures/pareto_front.pdf`
- `paper/figures/pareto_front.png`

## Paper table sources

- Table 1: `analysis/model_profile.py`, `experiments/compression/manifest.json`
- Table 2: `experiments/tinytcn/runs/fp32_100hz/test_report.txt`
- Table 3: `experiments/compression/phase_degradation.json`
- Pareto figure: `paper/figures/pareto_front.pdf`
- ESP32 table: hardware benchmark logs only, not PyTorch estimates

## ESP32-C3 measurement boundary

Do not fill ESP32 latency, SRAM, or power from PyTorch runs. Those values require:

- exportable model artifact
- TFLite Micro or equivalent firmware integration
- fixed 0.5 s input window
- measured on-device latency over repeated windows
- measured or logged peak SRAM
- measured current and fixed supply voltage for power
