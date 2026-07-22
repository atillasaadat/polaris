#ifndef POLARIS_SIM_ACTUATORS_MAGNETORQUER_HPP
#define POLARIS_SIM_ACTUATORS_MAGNETORQUER_HPP

/// @file
/// @brief Magnetorquer (magnetic torque rod) truth model (design doc §7).
///
/// Commanded-dipole-in → actual-dipole-out, per axis (a three-rod set, one rod
/// per body axis). The produced dipole differs from the command through three
/// real effects of a ferromagnetic-core rod:
///  - **Dipole limit**: the command saturates at the rod's rated moment.
///  - **Linearity error**: a scale-factor error across the operating range (the
///    NSS Taurus datasheet quotes ±5%).
///  - **Residual moment + hysteresis**: the core's B-H loop means the moment lags
///    the command and, when de-energized, retains a remanent moment. Both come
///    from one mechanism, modelled as a play (backlash) operator whose half-width
///    is the residual moment: commanding zero after saturating leaves exactly the
///    residual, and the moment "sticks" until the command moves past the loop.
///
/// The rod only produces a **dipole**; the resulting torque τ = m × B is computed
/// by the environment's magnetic-field path (`ResidualDipoleTorque`, §5.2), so
/// this model is deliberately field-agnostic. Power scales with the square of the
/// dipole (I²R, dipole ∝ current).
///
/// Fault injection is first-class (§9): stuck-on holds the last command; a dropout
/// collapses the moment to the (uncontrollable) residual.

#include <cmath>
#include <Eigen/Core>
#include <map>
#include <string>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::sim::actuators {

/// Magnetorquer parameters, SI. Per-axis identical (a symmetric three-rod set).
struct MagnetorquerSpec {
  double max_dipole_am2 = 0.0;       ///< rated magnetic moment per rod
  double residual_dipole_am2 = 0.0;  ///< remanent moment = hysteresis half-width
  double linearity = 0.0;            ///< scale-factor error, fractional (±0.05 bound)
  double power_max_w = 0.0;          ///< power at full dipole (scales with dipole²)

  /// Build a spec from catalog params (the keys used by the
  /// `config/hardware/magnetorquer/*.yaml` entries). See magnetorquer.cpp.
  static MagnetorquerSpec fromParams(const std::map<std::string, double>& params);
};

/// A three-axis magnetorquer set. Each body axis is one rod with its own
/// hysteresis state, so the residual/hysteresis history is tracked per axis.
class Magnetorquer {
 public:
  explicit Magnetorquer(const MagnetorquerSpec& spec) : spec_(spec) {}

  /// Command a body-frame dipole [A·m²] and return the actual dipole the rods
  /// produce, advancing the per-axis hysteresis state.
  math::Vec3<math::frames::Body> commandDipole(const math::Vec3<math::frames::Body>& dipole_cmd);

  /// Bus power [W] for the last produced dipole (∝ dipole²).
  double busPower() const;

  const math::Vec3<math::frames::Body>& dipole() const { return actual_; }

  // --- Fault injection (§9) --------------------------------------------------

  /// Stuck-on: ignore new commands, hold the last produced dipole until cleared.
  void setStuckOn(bool stuck) { fault_stuck_ = stuck; }

  /// Dropout: collapse the controllable moment; only the residual remains.
  void setDropout(bool dropped) { fault_dropout_ = dropped; }

  void clearFaults() {
    fault_stuck_ = false;
    fault_dropout_ = false;
  }

 private:
  /// Play (backlash) hysteresis on one axis: y = clamp(y_prev, x-r, x+r), with the
  /// result clamped to the rated moment. The remanence when x=0 is ±r.
  double applyAxis(double cmd, double& state) const;

  MagnetorquerSpec spec_;
  math::Vec3<math::frames::Body> actual_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d hysteresis_state_ = Eigen::Vector3d::Zero();
  bool fault_stuck_ = false;
  bool fault_dropout_ = false;
};

namespace catalog {

/// NewSpace Systems **Taurus** magnetorquer rod (datasheet v26.10, 2025). The rod
/// is a product family (0.2–400 A·m², ±5% linearity, <1.5 A·m² residual, 0.1–24 W);
/// this returns a representative unit sized by @p max_dipole_am2.
MagnetorquerSpec nssTaurus(double max_dipole_am2 = 30.0);

/// A generic magnetorquer rod — round numbers to copy and refine.
MagnetorquerSpec genericMagnetorquer();

}  // namespace catalog

}  // namespace polaris::sim::actuators

#endif  // POLARIS_SIM_ACTUATORS_MAGNETORQUER_HPP
