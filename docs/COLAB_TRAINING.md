# Colab Training Guide

Train all gait-phase baselines on Google Colab using the **shared fair-comparison protocol**.

## Prerequisites

- NONAN GaitPrint preprocessed data at `dataset/Xy` (621 trial CSVs)
- GPU runtime recommended (CPU works but is slow)

## 1. Clone and install

```python
!git clone https://github.com/Nontakorn-dev/DernDee_Research.git
%cd DernDee_Research
!pip install -r shared/requirements.txt
```

## 2. Mount dataset from Google Drive

Upload or symlink your preprocessed `Xy` folder to Drive, then:

```python
from google.colab import drive
drive.mount('/content/drive')

DATA_ROOT = "/content/drive/MyDrive/Dataset/processed/Xy"  # adjust path
```

Verify labels:

```bash
!python shared/scripts/check_xy_labels.py --channels bilateral --xy-root "$DATA_ROOT"
```

## 3. Train one model

Use `--lazy` on Colab to reduce RAM usage:

```bash
!python experiments/tinytcn/train.py \
  --config shared/configs/train_fair_comparison.json \
  --data-root /content/drive/MyDrive/Dataset/processed/Xy \
  --lazy \
  --device cuda
```

Other models:

- `experiments/cnn1d/train.py`
- `experiments/lstm_gru/train.py`
- `experiments/tcn/train.py`
- `experiments/transformer/train.py`

## 4. Train all models

```bash
!bash experiments/scripts/train_all_colab.sh \
  --config shared/configs/train_fair_comparison.json \
  --data-root /content/drive/MyDrive/Dataset/processed/Xy \
  --lazy \
  --device cuda
```

## 5. Outputs

Each model writes to `experiments/<model>/runs/fp32_100hz/`:

| File | Description |
|------|-------------|
| `best_model.pt` | Best validation macro F1 checkpoint |
| `norm_stats.json` | Train-set normalization |
| `history.json` | Per-epoch train loss + val F1 |
| `val_report.txt` / `test_report.txt` | Classification reports |
| `val_confusion_matrix.json` / `test_confusion_matrix.json` | Confusion matrices |
| `val_metrics.json` / `test_metrics.json` | Structured metrics |
| `config.json` | Shared protocol config used |

## 6. Download results

```python
!zip -r fp32_runs.zip experiments/*/runs/fp32_100hz
from google.colab import files
files.download("fp32_runs.zip")
```

## Fair-comparison contract

All models share the same settings from `shared/configs/train_fair_comparison.json`:

- 100 Hz, 0.5 s causal windows (50 samples)
- Train stride 5, val/test stride 1
- Subject split: 24 train / 5 val / 6 test
- Bilateral 12-channel IMU input, left-foot label (`phase_lt`)
- Adam lr=1e-3, batch 512, 50 epochs
- Balanced class weights
- Best checkpoint selected by validation macro F1

Do **not** change these per model unless running an explicit ablation.

## Re-evaluate a checkpoint

```bash
python shared/eval_checkpoint.py \
  --model tinytcn \
  --checkpoint experiments/tinytcn/runs/fp32_100hz/best_model.pt \
  --data-root /path/to/Xy \
  --split test \
  --out experiments/tinytcn/runs/fp32_100hz/test_metrics.json
```
