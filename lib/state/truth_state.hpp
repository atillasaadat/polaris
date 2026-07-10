#ifndef POLARIS_STATE_TRUTH_STATE_HPP
#define POLARIS_STATE_TRUTH_STATE_HPP

/// @file
/// @brief The simulation ground-truth state `TruthState` (design doc §8.0).
///
/// `TruthState` is the sim plant's ground-truth analogue of `EstimatedState`
/// (`estimated_state.hpp`): the **same kinematic/dynamic fields** — TAI epoch,
/// attitude, body rate, ECI position/velocity — but **without** covariance,
/// validity flags, biases, or estimation mode, because truth has no uncertainty
/// and no estimator artifacts.
///
/// This is a deliberate **type distinction** (design doc §2.3, §8.0,
/// REQ-SYS-005): only the sim produces and consumes `TruthState`; flight code is
/// written against `EstimatedState` and is therefore *structurally* unable to
/// touch truth. That separation is what makes the sim meaningful — estimators
/// and FDIR are exercised against real model error, never handed the answer.
///
/// Plain aggregate, no heap/exceptions (flight-safe conventions, §3.6).
/// Versioned so truth logs and golden fixtures stay readable (§22.1).

#include <cstdint>

#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "time/timescales.hpp"

namespace polaris::state {

/// Ground-truth vehicle state — sim-only (design doc §8.0). Same kinematics as
/// `EstimatedState`; no covariance/validity (truth is exact by construction).
struct TruthState {
  /// Schema version (§22.1). Bump on any field/layout/units change.
  static constexpr std::uint16_t kSchemaVersion = 1;

  time::Tai epoch{};  ///< TAI time tag of this truth sample (§3.2)

  math::Quat<math::frames::Body, math::frames::ECI> attitude{};  ///< Body ← ECI (JPL, q0≥0)
  math::Vec3<math::frames::Body> body_rate{};                    ///< ω_Body [rad/s]

  math::Vec3<math::frames::ECI> position{};  ///< r [m], ECI
  math::Vec3<math::frames::ECI> velocity{};  ///< v [m/s], ECI
};

}  // namespace polaris::state

#endif  // POLARIS_STATE_TRUTH_STATE_HPP
