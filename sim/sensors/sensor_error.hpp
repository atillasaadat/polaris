#ifndef POLARIS_SIM_SENSORS_SENSOR_ERROR_HPP
#define POLARIS_SIM_SENSORS_SENSOR_ERROR_HPP

/// @file
/// @brief The standard 3-axis vector-sensor error stack (design doc §6.1).
///
/// Every vector sensor (magnetometer, sun-sensor direction, gyro rate) turns a
/// true body-frame quantity into a measurement by passing it through the same
/// chain of imperfections. Rather than re-implement that per sensor, the chain is
/// one reusable model applied in the conventional order:
///
///   measured = saturate( quantize( M·truth + bias + noise ) )
///
///  - **M** — a 3×3 scale-factor + misalignment / non-orthogonality matrix. The
///    diagonal carries per-axis scale error, the off-diagonal the axis
///    misalignment and cross-axis coupling. For a magnetometer this is exactly
///    the soft-iron matrix; a fixed mounting misalignment folds in here too.
///  - **bias** — a constant additive offset (the magnetometer's hard-iron term).
///  - **noise** — zero-mean Gaussian, per-axis 1σ, drawn from the sensor's own
///    seeded stream so it is reproducible from `{config, seed}` (§3.6).
///  - **quantize** — rounding to a finite LSB resolution.
///  - **saturate** — clamping to the sensor's range.
///
/// Each stage is a no-op at its identity setting (identity matrix, zero
/// bias/noise, zero resolution, zero range), so a sensor pays only for the terms
/// it configures. Latency, sample-rate buffering, and time-varying bias drift are
/// **not** here: the first two belong to the §2.4 macro-step buffer (not built
/// until sim/io), and drift is a stateful per-sensor process added with the gyro.
///
/// Sim-side: heap and Eigen dynamic types are fine, but this stays fixed-size.
///
/// References:
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §4 (sensor models). [markley2014]

#include <algorithm>
#include <cmath>
#include <Eigen/Core>

#include "random/rng.hpp"

namespace polaris::sim::sensors {

/// The §6.1 error stack for a 3-axis vector sensor. Plain-`Eigen::Vector3d` in and
/// out: the caller supplies truth already in the sensor's axes and tags the frame.
struct VectorErrorModel {
  /// (I + scale)·misalignment, applied to the true vector. Identity = perfect.
  Eigen::Matrix3d scale_misalignment = Eigen::Matrix3d::Identity();
  /// Constant additive bias (hard-iron), same units as the measurement.
  Eigen::Vector3d bias = Eigen::Vector3d::Zero();
  /// Per-axis Gaussian noise 1σ. Zero on an axis disables noise there.
  Eigen::Vector3d noise_std = Eigen::Vector3d::Zero();
  /// LSB quantization step (0 disables). Same units as the measurement.
  double resolution = 0.0;
  /// Per-axis saturation limit |·| (0 disables).
  double range = 0.0;

  /// Apply the full chain. @p rng is advanced once per axis **iff** any axis has
  /// noise, so whether a given axis is noisy does not shift another axis's draw —
  /// keeping the stream position stable as the noise config changes.
  Eigen::Vector3d apply(const Eigen::Vector3d& truth, random::SplitMix64& rng) const {
    Eigen::Vector3d y = scale_misalignment * truth + bias;

    if ((noise_std.array() != 0.0).any()) {
      for (int i = 0; i < 3; ++i) {
        y[i] += noise_std[i] * rng.gaussian();
      }
    }
    if (resolution > 0.0) {
      for (int i = 0; i < 3; ++i) {
        y[i] = std::round(y[i] / resolution) * resolution;
      }
    }
    if (range > 0.0) {
      for (int i = 0; i < 3; ++i) {
        y[i] = std::clamp(y[i], -range, range);
      }
    }
    return y;
  }
};

}  // namespace polaris::sim::sensors

#endif  // POLARIS_SIM_SENSORS_SENSOR_ERROR_HPP
