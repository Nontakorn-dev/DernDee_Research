# Hybrid LSTM+GRU baseline

12-channel bilateral IMU → 4-phase gait classification using LSTM then GRU.

```bash
bash scripts/train.sh --config shared/configs/train_fair_comparison.json
```

Output: `runs/fp32_100hz/`
