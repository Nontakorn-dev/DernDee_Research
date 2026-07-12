/*
 * TCN (StandardTCN) synthetic benchmark for ESP32-C3 (no IMU/BLE).
 *
 * Prerequisites (Arduino IDE):
 *   - Board: ESP32C3 Dev Module
 *   - Library: Chirale_TensorFlowLite (TensorFlow Lite Micro)
 *
 * Before flashing, run on host:
 *   bash experiments/esp32/scripts/export_all.sh
 *
 * Rebuild with a different compression config by setting ACTIVE_CONFIG, e.g.:
 *   ACTIVE_CONFIG=Prune50 bash experiments/esp32/scripts/export_all.sh
 */

#include <Arduino.h>
#include "benchmark_config.h"
#include "norm_stats.h"
#include "test_window.h"
#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr int kInputFlatSize = kWindowSize * kNumChannels;
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

void normalize_window(const float raw[kWindowSize][kNumChannels], float out[kWindowSize][kNumChannels]) {
  for (int t = 0; t < kWindowSize; ++t) {
    for (int c = 0; c < kNumChannels; ++c) {
      out[t][c] = (raw[t][c] - kNormMean[c]) / kNormStd[c];
    }
  }
}

void fill_input_tensor(TfLiteTensor* input, const float normalized[kWindowSize][kNumChannels]) {
  if (input->type == kTfLiteFloat32) {
    float* dst = input->data.f;
    for (int c = 0; c < kNumChannels; ++c) {
      for (int t = 0; t < kWindowSize; ++t) {
        dst[c * kWindowSize + t] = normalized[t][c];
      }
    }
    return;
  }
  if (input->type == kTfLiteInt8) {
    const float scale = input->params.scale;
    const int zero_point = input->params.zero_point;
    int8_t* dst = input->data.int8;
    for (int c = 0; c < kNumChannels; ++c) {
      for (int t = 0; t < kWindowSize; ++t) {
        const float q = normalized[t][c] / scale + zero_point;
        int32_t v = static_cast<int32_t>(lroundf(q));
        if (v < -128) v = -128;
        if (v > 127) v = 127;
        dst[c * kWindowSize + t] = static_cast<int8_t>(v);
      }
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
      if (output->data.f[i] > best_val) {
        best_val = output->data.f[i];
        best = i;
      }
    }
    return best;
  }
  if (output->type == kTfLiteInt8) {
    int best = 0;
    int32_t best_val = output->data.int8[0];
    for (int i = 1; i < kOutputClasses; ++i) {
      if (output->data.int8[i] > best_val) {
        best_val = output->data.int8[i];
        best = i;
      }
    }
    return best;
  }
  return -1;
}

uint32_t percentile_us(uint32_t* samples, int count, float pct) {
  if (count <= 0) return 0;
  const int idx = min(count - 1, max(0, static_cast<int>(pct * (count - 1))));
  for (int i = 0; i < count; ++i) {
    for (int j = i + 1; j < count; ++j) {
      if (samples[j] < samples[i]) {
        const uint32_t tmp = samples[i];
        samples[i] = samples[j];
        samples[j] = tmp;
      }
    }
  }
  return samples[idx];
}

}  // namespace

void setup() {
  // ESP32-C3: USB Serial needs time to enumerate; do not block on Serial.
  Serial.begin(115200);
  delay(3000);
  Serial.println();
  Serial.println("# boot ok");
  Serial.flush();

  Serial.println("# TCN synthetic benchmark");
  Serial.print("# config=");
  Serial.println(MODEL_CONFIG);
  Serial.print("# model_bytes=");
  Serial.println(g_model_data_len);
  Serial.print("# tensor_arena_bytes=");
  Serial.println(TENSOR_ARENA_SIZE);

  const tflite::Model* model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch.");
    while (true) delay(1000);
  }

  static tflite::MicroMutableOpResolver<7> resolver = build_resolver();
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed. Increase TENSOR_ARENA_SIZE in benchmark_config.h.");
    while (true) delay(1000);
  }

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  float normalized[kWindowSize][kNumChannels];
  normalize_window(kReplayWindowRaw, normalized);

  for (int i = 0; i < WARMUP_INVOCATIONS; ++i) {
    fill_input_tensor(input, normalized);
    interpreter.Invoke();
  }

  static uint32_t latencies[BENCHMARK_INVOCATIONS];
  uint32_t min_us = UINT32_MAX;
  uint32_t max_us = 0;
  uint64_t sum_us = 0;

  Serial.println("config,trial,latency_us,free_heap,min_free_heap");

  for (int trial = 0; trial < BENCHMARK_INVOCATIONS; ++trial) {
    fill_input_tensor(input, normalized);
    const uint32_t t0 = micros();
    if (interpreter.Invoke() != kTfLiteOk) {
      Serial.print("# invoke failed at trial ");
      Serial.println(trial);
      break;
    }
    const uint32_t dt = micros() - t0;
    latencies[trial] = dt;
    sum_us += dt;
    min_us = min(min_us, dt);
    max_us = max(max_us, dt);

    Serial.print(MODEL_CONFIG);
    Serial.print(",");
    Serial.print(trial);
    Serial.print(",");
    Serial.print(dt);
    Serial.print(",");
    Serial.print(ESP.getFreeHeap());
    Serial.print(",");
    Serial.println(ESP.getMinFreeHeap());
    if ((trial % 100) == 0) {
      Serial.flush();
    }
  }

  const int pred = argmax_output(output);
  const uint32_t mean_us = static_cast<uint32_t>(sum_us / BENCHMARK_INVOCATIONS);
  uint32_t p95_us = percentile_us(latencies, BENCHMARK_INVOCATIONS, 0.95f);

  Serial.println("# summary");
  Serial.print("# pred=");
  Serial.println(pred);
  Serial.print("# replay_label=");
  Serial.println(kReplayLabel);
  Serial.print("# latency_mean_us=");
  Serial.println(mean_us);
  Serial.print("# latency_p95_us=");
  Serial.println(p95_us);
  Serial.print("# latency_min_us=");
  Serial.println(min_us);
  Serial.print("# latency_max_us=");
  Serial.println(max_us);
  Serial.print("# free_heap=");
  Serial.println(ESP.getFreeHeap());
  Serial.print("# min_free_heap=");
  Serial.println(ESP.getMinFreeHeap());
}

void loop() {
  // Heartbeat if setup finished; press RESET to re-run benchmark.
  Serial.println("# idle (press RESET to rerun benchmark)");
  Serial.flush();
  delay(10000);
}
