#pragma once
// Active benchmark build configuration.

#define MODEL_CONFIG "INT8"
#define TENSOR_ARENA_SIZE (120 * 1024)
#define WARMUP_INVOCATIONS 20
#define BENCHMARK_INVOCATIONS 1000
