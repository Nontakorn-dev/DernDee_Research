# CNN-LSTM hybrid baseline

12-channel bilateral IMU → 4-phase gait classification using Conv1d feature extraction followed by LSTM.

```bash
bash scripts/train.sh --config shared/configs/train_fair_comparison.json
```

Output: `runs/fp32_100hz/`
