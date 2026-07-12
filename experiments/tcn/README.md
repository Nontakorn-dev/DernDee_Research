# Standard TCN (primary deployment model)

12-channel bilateral IMU → 4-phase gait classification using four dilated causal convolution blocks
(dilations 1, 2, 4, 8; hidden width 32; ~11K parameters).

```bash
bash scripts/train.sh --config shared/configs/train_fair_comparison.json
```

Output: `runs/fp32_100hz/` or `runs/fold{K}_seed{S}_fp32_100hz/` under k-fold.

Compression and ESP32 export use this model as the deployment backbone.
