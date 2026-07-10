#ifndef POLARIS_STATE_ESTIMATED_STATE_HPP
#define POLARIS_STATE_ESTIMATED_STATE_HPP

/// @file
/// @brief The onboard navigation product `EstimatedState` (design doc §8.0).
///
/// The GNC chain has exactly **one** definition of vehicle state, produced once
/// by the estimators (§8.1/§8.3) and consumed everywhere — guidance, control,
/// FDIR, telemetry (REQ-SYS-005). `EstimatedState` is that onboard product:
/// TAI-tagged, attitude + rates + position/velocity with **frame tags on every
/// vector** (§3.1 boundary typing), gyro/accel biases, the full error-state
/// **covariance**, per-field **validity flags**, and the active **estimation
/// mode**. Multi-frame telemetry (ECI/ECEF/Keplerian/geodetic/LVLH/RIC) is a
/// derived *view* of this one struct (REQ-SYS-013), never a stored copy.
///
/// The counterpart `TruthState` (`truth_state.hpp`) carries the same kinematics
/// without covariance/validity, so the truth-vs-onboard separation (§2.3) is a
/// **type distinction**: flight code consumes `EstimatedState` and is
/// structurally unable to consume truth.
///
/// Plain aggregate, no heap/exceptions (flight-safe, §3.6). Versioned so logs,
/// telemetry, and golden fixtures stay readable as the schema evolves (§22.1).

#include <cstdint>
#include <Eigen/Core>

#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "time/timescales.hpp"

namespace polaris::state {

/// Active estimation mode (design doc §8.1). Ordered by increasing fidelity so
/// `>=` comparisons read naturally (e.g. "at least Coarse").
enum class EstimationMode : std::uint8_t {
  Invalid = 0,  ///< no valid solution (cold start / fault / diverged)
  Coarse = 1,   ///< sun-sensor + magnetometer + IMU coarse attitude (safe/acq, §8.1)
  Fine = 2,     ///< MEKF fine solution (nominal, §8.1)
};

/// Per-field validity of an `EstimatedState`. A consumer must check the relevant
/// flag before trusting a field: e.g. `Coarse` mode leaves position/velocity
/// invalid until the orbit filter (§8.3) converges. Defaults to all-invalid.
struct StateValidity {
  bool attitude{false};    ///< `attitude` is trustworthy
  bool body_rate{false};   ///< `body_rate` is trustworthy
  bool position{false};    ///< `position` is trustworthy
  bool velocity{false};    ///< `velocity` is trustworthy
  bool gyro_bias{false};   ///< `gyro_bias` estimate has converged
  bool accel_bias{false};  ///< `accel_bias` estimate has converged
  bool covariance{false};  ///< `covariance` is populated and meaningful
};

/// Row/column offsets of the 15-element MEKF error state within `Covariance`.
/// Block order — attitude, gyro bias, position, velocity, accel bias — is fixed
/// here so serialized covariances remain interpretable across builds (§22.1).
struct ErrorState {
  static constexpr int kAttitude = 0;    ///< δθ, 3 elements [rad] (Body attitude error)
  static constexpr int kGyroBias = 3;    ///< δb_g, 3 elements [rad/s]
  static constexpr int kPosition = 6;    ///< δr, 3 elements [m] (ECI)
  static constexpr int kVelocity = 9;    ///< δv, 3 elements [m/s] (ECI)
  static constexpr int kAccelBias = 12;  ///< δb_a, 3 elements [m/s^2]
  static constexpr int kDim = 15;        ///< total error-state dimension
};

/// 15×15 error-state covariance, blocked per `ErrorState`. Symmetric positive
/// semi-definite when valid; units follow each block's error state.
using Covariance = Eigen::Matrix<double, ErrorState::kDim, ErrorState::kDim>;

/// The single onboard navigation product (design doc §8.0, REQ-SYS-005). All
/// physical quantities carry SI units and frame tags; see field comments.
struct EstimatedState {
  /// Schema version (§22.1). Bump on any field/layout/units change.
  static constexpr std::uint16_t kSchemaVersion = 1;

  time::Tai epoch{};  ///< TAI time tag of this estimate (§3.2)

  math::Quat<math::frames::Body, math::frames::ECI> attitude{};  ///< Body ← ECI (JPL, q0≥0)
  math::Vec3<math::frames::Body> body_rate{};                    ///< ω_Body [rad/s]

  math::Vec3<math::frames::ECI> position{};  ///< r [m], canonical in ECI
  math::Vec3<math::frames::ECI> velocity{};  ///< v [m/s], canonical in ECI

  math::Vec3<math::frames::Body> gyro_bias{};   ///< estimated gyro bias [rad/s]
  math::Vec3<math::frames::Body> accel_bias{};  ///< estimated accel bias [m/s^2]

  Covariance covariance{Covariance::Zero()};     ///< 15×15 error-state (see ErrorState)
  StateValidity valid{};                         ///< per-field validity (default all false)
  EstimationMode mode{EstimationMode::Invalid};  ///< active estimator mode (§8.1)
};

}  // namespace polaris::state

#endif  // POLARIS_STATE_ESTIMATED_STATE_HPP
