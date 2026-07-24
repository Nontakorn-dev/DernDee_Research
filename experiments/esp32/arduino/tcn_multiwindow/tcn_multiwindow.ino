/*
 * TCN on-device vs. desktop-interpreter prediction cross-check (ESP32-C3).
 *
 * Runs the SAME exported .tflite artifact used for the latency benchmark
 * (tcn_benchmark) through TensorFlow Lite Micro on real hardware, over many
 * (~500-600) test windows sampled across every test-split subject/trial, and
 * prints one CSV row per window. Compare this Serial capture against the
 * desktop ai_edge_litert interpreter's predictions on the IDENTICAL windows
 * (saved in multiwindow_reference.npz) using:
 *
 *   python3 experiments/esp32/scripts/compare_ondevice_predictions.py \
 *       --serial-log <captured_output.csv>
 *
 * This answers: does the on-device TFLM runtime agree with the desktop
 * interpreter used everywhere else in the paper (Table III), not just on one
 * smoke-test window (host_validation.json) but across hundreds spanning all
 * four gait phases?
 *
 * Prerequisites (Arduino IDE):
 *   - Board: ESP32C3 Dev Module
 *   - Partition scheme: "Huge APP (3MB No OTA/1MB SPIFFS)" if the default
 *     scheme fails to fit (multiwindow_data.h is ~1.5 MB of flash constants).
 *   - Library: Chirale_TensorFlowLite (TensorFlow Lite Micro)
 *
 * Before flashing, generate the header for the config you want to test:
 *   python3 experiments/esp32/scripts/export_multiwindow_header.py \
 *       --active-config INT8 --num-windows 630
 *   cp experiments/esp32/arduino/tcn_benchmark/model_data_int8.h model_data.h
 * (repeat with --active-config INT8+Prune50, model_data_int8_plusprune50.h, etc.)
 *
 * After flashing: open Serial Monitor / Serial Plotter at 115200 baud, let it
 * run to "# done", then copy everything from the CSV header line
 * ("idx,label,pred,match,latency_us") through the final summary line into a
 * text file and pass that file to compare_ondevice_predictions.py.
 */

#include <Arduino.h>
#include "multiwindow_config.h"
#include "multiwindow_data.h"
#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr int kOutputClasses = 4;

alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];

tflite::MicroMutableOpResolver<7> build_resolver() {
  tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddTranspose();
  resolver.AddReshape();
  resolver.AddPad();
  resolver.AddConv2D();
  resolver.AddAdd();
  resolver.AddMean();
  resolver.AddFullyConnected();
  return resolver;
}

void fill_input_tensor(TfLiteTensor* input, const float* normalized_flat) {
  if (input->type == kTfLiteFloat32) {
    for (int i = 0; i < kMultiWindowFlatSize; ++i) {
      input->data.f[i] = normalized_flat[i];
    }
    return;
  }
  if (input->type == kTfLiteInt8) {
    const float scale = input->params.scale;
    const int zero_point = input->params.zero_point;
    for (int i = 0; i < kMultiWindowFlatSize; ++i) {
      const float q = normalized_flat[i] / scale + zero_point;
      int32_t v = static_cast<int32_t>(lroundf(q));
      if (v < -128) v = -128;
      if (v > 127) v = 127;
      input->data.int8[i] = static_cast<int8_t>(v);
    }
    return;
  }
  Serial.println("Unsupported input tensor type.");
}

int argmax_output(const TfLiteTensor* output) {
  if (output->type == kTfLiteFloat32) {
    int best = 0;
    float best_val = output->data.f[0];
    for (int i = 1; i < kOutputClasses; ++i) {
      if (output->data.f[i] > best_val) { best_val = output->data.f[i]; best = i; }
    }
    return best;
  }
  if (output->type == kTfLiteInt8) {
    int best = 0;
    int32_t best_val = output->data.int8[0];
    for (int i = 1; i < kOutputClasses; ++i) {
      if (output->data.int8[i] > best_val) { best_val = output->data.int8[i]; best = i; }
    }
    return best;
  }
  return -1;
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(3000);
  Serial.println();
  Serial.println("# boot ok");
  Serial.print("# config=");
  Serial.println(MODEL_CONFIG);
  Serial.print("# num_windows=");
  Serial.println(kNumMultiWindows);
  Serial.flush();

  const tflite::Model* model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch.");
    while (true) delay(1000);
  }

  static tflite::MicroMutableOpResolver<7> resolver = build_resolver();
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed. Increase TENSOR_ARENA_SIZE.");
    while (true) delay(1000);
  }

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  Serial.println("idx,label,pred,match,latency_us");

  int n_correct_vs_label = 0;
  uint64_t sum_us = 0;

  for (int i = 0; i < kNumMultiWindows; ++i) {
    fill_input_tensor(input, kMultiWindowNormalized[i]);
    const uint32_t t0 = micros();
    if (interpreter.Invoke() != kTfLiteOk) {
      Serial.print("# invoke failed at window ");
      Serial.println(i);
      continue;
    }
    const uint32_t dt = micros() - t0;
    sum_us += dt;

    const int pred = argmax_output(output);
    const int label = kMultiWindowLabels[i];
    const int match = (pred == label) ? 1 : 0;
    n_correct_vs_label += match;

    Serial.print(i);
    Serial.print(",");
    Serial.print(label);
    Serial.print(",");
    Serial.print(pred);
    Serial.print(",");
    Serial.print(match);
    Serial.print(",");
    Serial.println(dt);

    if ((i % 50) == 0) Serial.flush();
  }

  Serial.println("# summary");
  Serial.print("# n_windows=");
  Serial.println(kNumMultiWindows);
  Serial.print("# accuracy_vs_label=");
  Serial.println((float)n_correct_vs_label / kNumMultiWindows, 4);
  Serial.print("# mean_latency_us=");
  Serial.println((uint32_t)(sum_us / kNumMultiWindows));
  Serial.println("# done");
  Serial.flush();
}

void loop() {
  Serial.println("# idle (press RESET to rerun)");
  Serial.flush();
  delay(10000);
}
