# ESP32-S3 deployment

On-device inference and benchmarking protocol for compressed TinyTCN.

Do not fill paper latency, SRAM, or power tables from PyTorch runs. Those values require hardware measurement on ESP32-S3.

## Firmware (acquisition)

Parent repo: `DernDee_Robustness/Hardware/esp32_ICM20948_Ver2.ino`  
Streams 12ch IMU @ 100 Hz over BLE.

## Input contract

- sampling rate: 100 Hz
- input window: 50 samples (0.5 s)
- input channels: 12 bilateral shank IMU channels
- normalization: use the training-set `norm_stats.json` bundled with the selected model
- output: four logits/classes for LR, LS, PSw, Sw

## Accuracy source of truth

Before exporting to firmware, run:

```bash
bash experiments/compression/scripts/run_compression.sh
```

Use the best trade-off config from:

- `experiments/compression/manifest.json`
- `experiments/compression/phase_degradation.json`
- `paper/figures/pareto_front.pdf`

## Model deploy (TODO)

1. Export TinyTCN to TFLite Micro from `experiments/tinytcn/runs/fp32_100hz/best_model.pt`
2. Export selected compressed configs after PyTorch evaluation, starting with INT8 and INT8+Prune50
3. Flash the `.tflite` / generated C array to ESP32-S3 firmware
4. Feed normalized 50-sample windows through the same channel order used in training
5. Record latency (ms), peak SRAM (KB), and current draw for Table 4 in the paper

Target: inference &lt; 36 ms per 0.5 s window.

## Benchmark template

Record one row per model/config:

| Field | Meaning |
|-------|---------|
| config | FP32, INT8, Prune50, INT8+Prune50 |
| artifact | path or firmware build id |
| input_window_samples | should be 50 |
| trials | number of repeated inference windows |
| latency_mean_ms | mean inference time |
| latency_p95_ms | p95 inference time |
| peak_sram_kb | peak SRAM during inference |
| current_ma | measured current at fixed supply voltage |
| power_mw | current_ma x voltage |
| notes | firmware, board, clock, PSRAM setting |

Suggested output path once measurements exist:

```
experiments/esp32/benchmarks/esp32_s3_metrics.json
```
