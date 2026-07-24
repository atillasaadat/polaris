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

#include <vector>

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

/// A settable external wrench — the actuator-feedback channel of the §2.4
/// closed loop. The loop computes the net actuator effect on the body each
/// micro-interval (reaction-wheel reaction torques through the assembly's W,
/// magnetorquer m×B) and writes it here; the plant integrates it like any other
/// external model. Zero until set, so composing it into a run with no commands
/// changes nothing.
class CommandedWrench : public ForceTorqueModel {
 public:
  /// Set the wrench for the next propagation interval (zero-order hold).
  void set(const math::Vec3<math::frames::ECI>& accel,
           const math::Vec3<math::frames::Body>& torque) {
    accel_ = accel;
    torque_ = torque;
  }

  math::Vec3<math::frames::ECI> acceleration(const state::TruthState&) const override {
    return accel_;
  }

  math::Vec3<math::frames::Body> torque(const state::TruthState&) const override { return torque_; }

 private:
  math::Vec3<math::frames::ECI> accel_{};
  math::Vec3<math::frames::Body> torque_{};
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

/// Sum of several `ForceTorqueModel`s — the superposition the plant actually
/// flies (e.g. EGM2008 gravity + third-body + drag + SRP). Acceleration and
/// torque are both linear in the sources, so each is the sum of the parts.
///
/// Holds **non-owning** pointers; every added model must outlive the composite
/// (same lifetime rule the plant already imposes on its model). Sim-side, so heap
/// and `std::vector` are fine (`sim/CLAUDE.md`).
class CompositeForceModel : public ForceTorqueModel {
 public:
  /// Append a component. Null is ignored (a no-op source).
  void add(const ForceTorqueModel* model) {
    if (model != nullptr) {
      models_.push_back(model);
    }
  }

  std::size_t size() const { return models_.size(); }

  math::Vec3<math::frames::ECI> acceleration(const state::TruthState& s) const override {
    Eigen::Vector3d a = Eigen::Vector3d::Zero();
    for (const ForceTorqueModel* m : models_) {
      a += m->acceleration(s).eigen();
    }
    return math::Vec3<math::frames::ECI>(a);
  }

  math::Vec3<math::frames::Body> torque(const state::TruthState& s) const override {
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    for (const ForceTorqueModel* m : models_) {
      t += m->torque(s).eigen();
    }
    return math::Vec3<math::frames::Body>(t);
  }

 private:
  std::vector<const ForceTorqueModel*> models_;
};

}  // namespace polaris::sim::dynamics

#endif  // POLARIS_SIM_DYNAMICS_FORCE_TORQUE_HPP
