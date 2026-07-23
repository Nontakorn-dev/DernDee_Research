#pragma once
// Auto-generated from training norm_stats.json. Channel order matches IMU_INPUT_COLUMNS.

static const int kNumChannels = 12;
static const int kWindowSize = 50;

static const float kNormMean[kNumChannels] = {
  0.43052959f, 0.07009777f, -3.09720898f, 0.57073510f,
  5.93750620f, -7.21373081f, 0.48247010f, -0.27118099f,
  -3.54935503f, -1.55391896f, 5.78918314f, 7.84295321f,
};

static const float kNormStd[kNumChannels] = {
  642.46466064f, 519.63073730f, 457.11532593f, 75.19622803f,
  151.06283569f, 118.38034058f, 641.38220215f, 533.81140137f,
  457.05187988f, 74.47004700f, 157.84135437f, 111.89402771f,
};
