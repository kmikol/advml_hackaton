// Auto-generated normalization parameters
#pragma once
#include <cstddef>

namespace norm {
constexpr std::size_t kNumX = 4;
constexpr std::size_t kNumY = 1;

enum XFeatureIndex {
  X_DAY_TIME_PERIODIC = 0, // day_time_periodic
  X_YEAR_TIME_PERIODIC = 1, // year_time_periodic
  X_TEMPERATURE = 2, // Temperature
  X_HUMIDITY = 3, // Humidity
};

constexpr float kXMean[kNumX] = { 5.1647737775602522e-17, 0.0027472495222158281, 18.810023962148961, 68.25951846764346 };
constexpr float kXStd[kNumX]  = { 0.70710678118654757, 0.70612947304053464, 5.8154203644008682, 15.551028829814287 };
constexpr float kYMean[1] = { 32344.970563585928 };
constexpr float kYStd[1]  = { 7130.4945449241859 };

// Normalize inputs: out[i] = (in[i] - kXMean[i]) / max(kXStd[i], 1e-12)
inline void NormalizeX(const float in[kNumX], float out[kNumX]) {
  for (std::size_t i = 0; i < kNumX; ++i) {
    const float s = (kXStd[i] == 0.0f) ? 1.0f : kXStd[i];
    out[i] = (in[i] - kXMean[i]) / s;
  }
}

// Normalize a single Y (index 0); extend if you add more outputs
inline float NormalizeY(float y, std::size_t idx = 0) {
  const float s = (kYStd[idx] == 0.0f) ? 1.0f : kYStd[idx];
  return (y - kYMean[idx]) / s;
}

// Denormalize Y back to original units
inline float DenormalizeY(float y_norm, std::size_t idx = 0) {
  return y_norm * kYStd[idx] + kYMean[idx];
}

} // namespace norm
