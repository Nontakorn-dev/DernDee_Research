# TinyTCN

12-channel bilateral IMU model for 4-phase gait classification.

Current research contract:

- 100 Hz target rate
- 0.5 s causal windows (50 samples)
- 12 IMU channels
- hidden width 32
- about 11K parameters
- 50 training epochs (save best validation macro F1)
- labels: LR, LS, PSw, Sw

```bash
bash scripts/train.sh --config shared/configs/train_fair_comparison.json
```

Output: `runs/fp32_100hz/` (`best_model.pt`, reports, metrics JSON)

Uses `shared/splits/subject_split.csv`.

Evaluate an existing checkpoint with structured metrics:

```bash
python ../../shared/eval_checkpoint.py \
  --checkpoint runs/fp32_100hz/best_model.pt \
  --out runs/fp32_100hz/test_metrics.json
```
