#pragma once
// Active multi-window validation build configuration.
// Regenerate multiwindow_data.h / model_data.h for a different config via:
//   python3 experiments/esp32/scripts/export_multiwindow_header.py --active-config <CFG>
//   cp experiments/esp32/exports/<CFG>/model.tflite -> re-run export_headers-style copy,
//   or simply: cp experiments/esp32/arduino/tcn_benchmark/model_data_<cfg>.h model_data.h

#define MODEL_CONFIG "INT8"
#define TENSOR_ARENA_SIZE (120 * 1024)
