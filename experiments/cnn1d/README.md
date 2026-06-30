# 1D-CNN baseline

12-channel bilateral IMU → 4-phase gait classification using stacked Conv1d blocks.

```bash
bash scripts/train.sh --config shared/configs/train_fair_comparison.json
```

Output: `runs/fp32_100hz/`
