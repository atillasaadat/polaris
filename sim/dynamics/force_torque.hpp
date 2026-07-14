#ifndef POLARIS_SIM_DYNAMICS_FORCE_TORQUE_HPP
#define POLARIS_SIM_DYNAMICS_FORCE_TORQUE_HPP

/// @file
/// @brief External force/torque providers for the 6DOF plant (design doc §5.1).
///
/// A `ForceTorqueModel` returns the specific external **acceleration** (ECI,
/// [m/s^2]) and external **torque** (Body, [N·m]) acting on the vehicle at a
/// given truth state. Returning acceleration (not force) means gravity needs no
/// vehicle mass — the mass-independent case is the common one. Environment
/// models (EGM2008, drag, SRP, third-body, disturbance torques — REQ-SIM-002)
/// land as further implementations of this interface in a later push; the plant
/// integrates whatever provider it is handed.
///
/// Sim-side (`sim/CLAUDE.md`): virtual dispatch/heap are fine here.

#include "constants/constants.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "state/truth_state.hpp"

namespace polaris::sim::dynamics {

/// Source of external acceleration + torque on the rigid body.
class ForceTorqueModel {
 public:
  virtual ~ForceTorqueModel() = default;

  /// Specific external acceleration in ECI [m/s^2] at truth state @p s.
  virtual math::Vec3<math::frames::ECI> acceleration(const state::TruthState& s) const = 0;

  /// External torque in the Body frame [N·m] at truth state @p s.
  virtual math::Vec3<math::frames::Body> torque(const state::TruthState& s) const = 0;
};

/// Free drift: no external acceleration, no external torque. Torque-free
/// rotation and force-free translation — the analytic conservation baselines.
class NoForceModel : public ForceTorqueModel {
 public:
  math::Vec3<math::frames::ECI> acceleration(const state::TruthState&) const override {
    return math::Vec3<math::frames::ECI>::Zero();
  }

  math::Vec3<math::frames::Body> torque(const state::TruthState&) const override {
    return math::Vec3<math::frames::Body>::Zero();
  }
};

/// Point-mass (two-body) gravity: a = -mu r / |r|^3. Rotation is left torque-free
/// (a point mass exerts no torque). Default mu is Earth's WGS84 value.
class TwoBodyGravity : public ForceTorqueModel {
 public:
  explicit TwoBodyGravity(double mu = constants::wgs84::kGM) : mu_(mu) {}

  math::Vec3<math::frames::ECI> acceleration(const state::TruthState& s) const override {
    const Eigen::Vector3d r = s.position.eigen();
    const double rn = r.norm();
    // Boundary guard: a singular radius yields no finite acceleration; a real
    // orbit never reaches r = 0, so return zero rather than a NaN (§3.6).
    if (rn < kMinRadius_) {
      return math::Vec3<math::frames::ECI>::Zero();
    }
    return math::Vec3<math::frames::ECI>(Eigen::Vector3d(-mu_ * r / (rn * rn * rn)));
  }

  math::Vec3<math::frames::Body> torque(const state::TruthState&) const override {
    return math::Vec3<math::frames::Body>::Zero();
  }

  double mu() const { return mu_; }

 private:
  static constexpr double kMinRadius_ = 1.0;  ///< [m]
  double mu_;
};

}  // namespace polaris::sim::dynamics

#endif  // POLARIS_SIM_DYNAMICS_FORCE_TORQUE_HPP
