# ESP32-C3 synthetic benchmark (TinyTCN)

Hardware latency / SRAM measurements for compressed TinyTCN **without a real IMU**.
The sketch replays one fixed `(50, 12)` window, normalizes with training stats, and
runs TFLite Micro inference in a tight loop.

Primary configs: `INT8`, `INT8+Prune50`  
Comparison (if SRAM allows): `FP32`, `Prune50`  
Skipped: `INT4` (PyTorch accuracy floor only; TFLite Micro on ESP32 targets float32/int8)

## Pipeline

Export uses a dedicated Python 3.12 env (`.venv-export`) because TensorFlow does not support Python 3.14:

```bash
bash experiments/esp32/scripts/setup_export_env.sh
```

Bootstrap + export + headers + validation:

```bash
bash experiments/esp32/scripts/export_all.sh
```

Manual steps:

```bash
.venv/bin/python experiments/esp32/scripts/export_tflite.py
.venv/bin/python experiments/esp32/scripts/export_headers.py --active-config INT8
.venv/bin/python experiments/esp32/scripts/validate_tflite.py
```

## Arduino IDE setup

| Setting | Value |
|---------|-------|
| Board | **ESP32C3 Dev Module** |
| USB CDC On Boot | Enabled (for Serial on C3) |
| CPU frequency | 160 MHz (default) |
| Flash size | Match your module (e.g. 4 MB) |
| Partition scheme | Default |
| Library | **Arduino_TensorFlowLite** (Library Manager) |

Open sketch:

```
experiments/esp32/arduino/tinytcn_benchmark/tinytcn_benchmark.ino
```

### Switch compression config

Re-run header generation with a different active model, then re-flash:

```bash
ACTIVE_CONFIG=INT8+Prune50 TENSOR_ARENA_KB=120 bash experiments/esp32/scripts/export_all.sh
```

If `AllocateTensors()` fails, increase arena size:

```bash
TENSOR_ARENA_KB=160 ACTIVE_CONFIG=FP32 bash experiments/esp32/scripts/export_all.sh
```

Suggested progression on ESP32-C3 (~400 KB SRAM, usually no PSRAM): **80 → 120 → 160 → 200 KB**.

## Flash and capture

1. Connect ESP32-C3 via USB
2. Select the correct serial port
3. Upload sketch
4. Open Serial Monitor at **115200 baud**
5. Wait for `# summary` block

### Serial output format

Per-trial CSV rows:

```
config,trial,latency_us,free_heap,min_free_heap
INT8,0,12345,234567,198765
...
```

Summary footer:

```
# summary
# pred=1
# replay_label=1
# latency_mean_us=12000
# latency_p95_us=13500
# latency_min_us=11000
# latency_max_us=15000
# free_heap=...
# min_free_heap=...
```

Copy values into `experiments/esp32/benchmarks/esp32_c3_metrics.json` (convert µs → ms by dividing by 1000).

Repeat for each config (`INT8`, `INT8+Prune50`, then `FP32` / `Prune50` if they fit).

## Input contract

- Sampling rate: 100 Hz (synthetic replay only)
- Window: 50 samples × 12 channels
- Channel order (`shared/gait_labels.py` / `IMU_INPUT_COLUMNS`):
  `lt_acc_x/y/z, lt_gyr_x/y/z, rt_acc_x/y/z, rt_gyr_x/y/z`
- Normalization: training-set `norm_stats.json` → `norm_stats.h`
- TFLite input layout: **NCHW** `[1, 12, 50]` (firmware transposes normalized `(50, 12)` window)
- Output: 4 logits → argmax over `{LR, LS, PSw, Sw}`

## Host validation

Before flashing, confirm PyTorch vs TFLite on one fixed replay window:

```bash
.venv/bin/python experiments/esp32/scripts/validate_tflite.py
```

Results: `experiments/esp32/benchmarks/host_validation.json`

INT8 may differ slightly from PyTorch FP32 logits; smoke test passes on **matching argmax** with config-specific logit tolerance (FP32/Prune50 ≤ 0.05, INT8 variants ≤ 1.0).

## JSON result format (`esp32_c3_metrics.json`)

```json
{
  "board": "ESP32-C3 Dev Module",
  "toolchain": "Arduino IDE",
  "configs": {
    "INT8": {
      "artifact": "experiments/esp32/exports/INT8/model.tflite",
      "input_window_samples": 50,
      "trials": 1000,
      "warmup": 20,
      "tensor_arena_kb": 120,
      "latency_mean_ms": 12.0,
      "latency_p95_ms": 13.5,
      "latency_min_ms": 11.0,
      "latency_max_ms": 15.0,
      "free_heap_bytes": 234567,
      "min_free_heap_bytes": 198765,
      "pred": 1,
      "replay_label": 1
    }
  }
}
```

## Notes

- PyTorch compression metrics (`experiments/compression/manifest.json`) measure **accuracy**; this folder measures **on-device latency/SRAM**.
- `INT8` / `INT8+Prune50` TFLite exports use **full-integer PTQ** from float weights, not the PyTorch QDQ proxy checkpoints.
- If FP32 float models exhaust SRAM on ESP32-C3, report that only INT8 variants fit and note ESP32-S3 as the deployment target from the paper abstract.
- Target inference budget: **< 36 ms** per 0.5 s window.

## Related paths

| Path | Purpose |
|------|---------|
| `experiments/esp32/scripts/export_tflite.py` | PyTorch → ONNX → TFLite |
| `experiments/esp32/scripts/export_headers.py` | C headers for firmware |
| `experiments/esp32/scripts/validate_tflite.py` | Host smoke test |
| `experiments/esp32/arduino/tinytcn_benchmark/` | Benchmark sketch |
| `experiments/esp32/benchmarks/esp32_c3_metrics.json` | Hardware results template |
| `experiments/compression/manifest.json` | Accuracy / size source of truth |

## Acquisition firmware (real IMU, optional)

Parent repo reference for 12-ch @ 100 Hz BLE streaming:
`DernDee_Robustness/Hardware/esp32_ICM20948_Ver2.ino`

This synthetic benchmark does **not** use that firmware.
