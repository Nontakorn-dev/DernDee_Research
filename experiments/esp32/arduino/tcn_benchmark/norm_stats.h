#pragma once
// Auto-generated from training norm_stats.json. Channel order matches IMU_INPUT_COLUMNS.

static const int kNumChannels = 12;
static const int kWindowSize = 50;

static const float kNormMean[kNumChannels] = {
  0.21901543f, 0.36085039f, -3.18885875f, 0.39628676f,
  5.58511496f, -7.39538336f, 0.37378219f, 0.08577938f,
  -3.63241410f, -1.48162782f, 5.18440628f, 7.23658943f,
};

static const float kNormStd[kNumChannels] = {
  643.23022461f, 514.16033936f, 457.08792114f, 71.44033813f,
  155.14262390f, 114.43163300f, 655.43084717f, 543.21783447f,
  469.85620117f, 72.63809204f, 159.88410950f, 109.49715424f,
};
