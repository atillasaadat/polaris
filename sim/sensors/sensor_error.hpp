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
///    seeded stream so it is reproducible from `{config, seed}` (§3.5).
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
///
/// Implements the shared error stack of REQ-SIM-003.

#include <algorithm>
#include <cmath>
#include <Eigen/Core>

#include "random/rng.hpp"

namespace polaris::sim::sensors {

/// The §6.1 error stack for a 3-axis vector sensor. Plain-`Eigen::Vector3d` in and
/// out: the caller supplies truth already in the sensor's axes and tags the frame.
///
/// **Error model.** For a true vector \f$x\f$, the measurement is the composition
/// applied in `apply()` in this exact order:
/// \f[
///   y = \operatorname{sat}_{R}\!\Big(\, Q_{\delta}\big(\, M x + b + n \,\big) \Big),
///   \qquad n_i \sim \mathcal{N}\!\big(0,\ \sigma_i^2\big),
/// \f]
/// where the four configurable stages are, each a no-op at its identity setting:
///  - \f$M\f$ (`scale_misalignment`) — the \f$3\times3\f$ scale-factor +
///    misalignment / soft-iron matrix; \f$M = I\f$ is perfect.
///  - \f$b\f$ (`bias`) — the constant additive (hard-iron) offset.
///  - \f$n\f$ — zero-mean Gaussian noise with per-axis \f$\sigma_i\f$ (`noise_std`);
///    an axis with \f$\sigma_i = 0\f$ draws nothing there.
///  - \f$Q_{\delta}(v) = \delta\,\operatorname{round}(v/\delta)\f$ — LSB
///    quantization with step \f$\delta\f$ (`resolution`); \f$\delta = 0\f$ disables it.
///  - \f$\operatorname{sat}_{R}(v) = \operatorname{clamp}(v,\,-R,\,+R)\f$ — per-axis
///    saturation to the range \f$R\f$; \f$R = 0\f$ disables it.
///
/// When `noise_enabled` is false the whole stack collapses to the identity
/// \f$y = x\f$ (an **ideal** sensor: no scale, misalignment, bias, noise,
/// quantization, or saturation), used for noise-free baseline runs (§6.2).
///
/// Markley & Crassidis §4 [markley2014].
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
  /// Master switch (§6.2). False makes the sensor **ideal** — the measurement is
  /// the truth, with no scale, misalignment, bias, noise, quantization, or
  /// saturation — for runs that need a noise-free baseline (bring-up, algorithm
  /// bring-up, with/without-noise comparisons). It disables the whole stack, not
  /// only the random term, so an ideal sensor also has no fixed bias.
  bool noise_enabled = true;

  /// Apply the full chain. @p rng is advanced once per axis **iff** any axis has
  /// noise, so whether a given axis is noisy does not shift another axis's draw —
  /// keeping the stream position stable as the noise config changes.
  ///
  /// @warning The draw is conditioned on `noise_std`, which is **fixed at
  /// construction** for the whole run — that is what makes the conditional safe.
  /// If a future change ever makes `noise_std` runtime-mutable (e.g. a fault
  /// that zeroes one axis mid-run), this must switch to the always-draw pattern
  /// used elsewhere (`drawGaussian`, the star tracker's fixed nine draws), or
  /// reproducibility silently breaks (§3.5).
  Eigen::Vector3d apply(const Eigen::Vector3d& truth, random::SplitMix64& rng) const {
    if (!noise_enabled) {
      return truth;  // ideal sensor: measurement is the truth
    }
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
