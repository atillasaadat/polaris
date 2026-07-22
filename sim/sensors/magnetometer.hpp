#ifndef POLARIS_SIM_SENSORS_MAGNETOMETER_HPP
#define POLARIS_SIM_SENSORS_MAGNETOMETER_HPP

/// @file
/// @brief Three-axis magnetometer truth model (design doc §6.2).
///
/// Truth-in → measurement-out: given the ambient magnetic field expressed in the
/// body frame (the caller rotates the IGRF field, §6.2/§5.2, from ECI through the
/// vehicle attitude), it produces the field a real magnetometer would report —
/// with hard-iron bias, soft-iron + misalignment, and noise, via the shared §6.1
/// error stack. The FSW pairs this measurement with its onboard IGRF-14 modelled
/// field for measured-vs-modelled checks and coarse attitude (§8.1).
///
/// **Fault injection is first-class** (sim/CLAUDE.md; §9): the FDIR integration
/// suite drives a bias jump or a dropout through the hooks here, so a scenario can
/// exercise the health monitors without a special build.
///
/// Each instance owns an independent, seed-derived noise stream (§182), so it is
/// bit-reproducible from `{config, seed}` and adding another sensor does not
/// perturb it.

#include <Eigen/Core>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "random/rng.hpp"
#include "sensors/sensor_error.hpp"
#include "time/timescales.hpp"

namespace polaris::sim::sensors {

/// One magnetometer sample: the measured field in body axes, a validity flag, and
/// the truth time it was taken (the §2.4 buffer will add read-time latency later).
struct MagnetometerMeasurement {
  math::Vec3<math::frames::Body> field_tesla{};
  bool valid{true};
  time::Tai time_tag{};
};

/// A three-axis magnetometer. Construct with its error stack and a per-source
/// stream id under the run's master seed.
class Magnetometer {
 public:
  Magnetometer(const VectorErrorModel& error, std::uint64_t master_seed, std::uint64_t stream_id)
      : error_(error), rng_(random::streamRng(master_seed, stream_id)) {}

  /// Measure the true body-frame field @p b_truth_body at truth time @p epoch.
  MagnetometerMeasurement sample(const time::Tai& epoch,
                                 const math::Vec3<math::frames::Body>& b_truth_body);

  // --- Fault injection (§9) --------------------------------------------------

  /// Set a persistent bias jump [T] until cleared — a step fault the health
  /// monitors must catch. Replaces any previous jump (it does not accumulate).
  /// The offset is applied with the hard-iron bias, so it is subject to the
  /// sensor's quantization and range saturation just like a real one.
  void injectBiasJump(const math::Vec3<math::frames::Body>& delta_tesla) {
    fault_bias_ = delta_tesla.eigen();
  }

  /// Force every subsequent sample invalid (a sensor dropout) until cleared.
  void setDropout(bool dropped) { fault_dropout_ = dropped; }

  /// Clear all injected faults, returning the sensor to nominal.
  void clearFaults() {
    fault_bias_.setZero();
    fault_dropout_ = false;
  }

 private:
  VectorErrorModel error_;
  random::SplitMix64 rng_;
  Eigen::Vector3d fault_bias_ = Eigen::Vector3d::Zero();
  bool fault_dropout_ = false;
};

}  // namespace polaris::sim::sensors

#endif  // POLARIS_SIM_SENSORS_MAGNETOMETER_HPP
