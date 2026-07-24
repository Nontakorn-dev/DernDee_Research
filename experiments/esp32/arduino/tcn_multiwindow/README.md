# On-device TFLM vs. desktop-interpreter cross-check

This sketch answers a specific reviewer question: everywhere else in the paper,
"real deployed accuracy" is measured by running the exported `.tflite` file
through the **desktop** `ai_edge_litert` interpreter (Table III). The actual
ESP32-C3 firmware runs the same file through **TensorFlow Lite Micro**, a
different runtime implementation. Do the two agree, beyond the single
one-window smoke test already in `host_validation.json`?

## What's already generated

- `multiwindow_data.h` -- 630 test windows (already normalized), sampled evenly
  across all 126 test-split trials for fold0/seed42, covering all four gait
  phases (LR/LS/PSw/Sw). Config: **INT8**.
- `model_data.h` -- the INT8 `.tflite` bytes (same file used for Table III/V).
- `multiwindow_reference.npz` / `multiwindow_meta.json` -- host-side copies of
  the identical windows + labels, so the comparison script never has to
  re-derive the sample (byte-for-byte the same input the board sees).

## Steps

1. Open `tcn_multiwindow.ino` in Arduino IDE.
   - Board: **ESP32C3 Dev Module**
   - Partition scheme: if you get a "sketch too big" error, switch to
     **Huge APP (3MB No OTA/1MB SPIFFS)** -- the embedded window data is
     ~1.5 MB of flash constants.
   - Library: Chirale_TensorFlowLite (same one used for `tcn_benchmark`).
2. Flash it. Open Serial Monitor at 115200 baud.
3. Let it run to `# done` (630 windows at roughly the per-config inference
   latency from Table V -- a few minutes for INT8).
4. Select and copy everything from the `idx,label,pred,match,latency_us`
   header line through `# done` into a text file, e.g. `serial_capture.txt`.
5. From the repo root:
   ```
   .venv-export/bin/python3 experiments/esp32/scripts/compare_ondevice_predictions.py \
       --serial-log serial_capture.txt --active-config INT8
   ```
   This reports the on-device (TFLM) vs. desktop (`ai_edge_litert`) argmax
   **agreement rate** across all 630 windows, plus both sides' accuracy vs.
   ground truth, and lists any individual disagreements.

## To test a different config (e.g. INT8+Prune50)

```
.venv-export/bin/python3 experiments/esp32/scripts/export_multiwindow_header.py \
    --active-config INT8+Prune50 --num-windows 630
cp experiments/esp32/arduino/tcn_benchmark/model_data_int8_plusprune50.h model_data.h
```
Then edit `multiwindow_config.h`'s `MODEL_CONFIG` string to match (cosmetic,
only affects the boot banner), reflash, and re-run the comparison script with
`--active-config INT8+Prune50`.

## What a "pass" looks like

Agreement should be at or near 100%: the TFLite file format is the same
bytes either way, and both runtimes implement the same quantized-int8 kernel
semantics. A material disagreement rate (not just 1-2 borderline windows)
would indicate a genuine TFLM-vs-desktop kernel discrepancy worth
investigating before trusting Table III's numbers as "deployed" accuracy.
