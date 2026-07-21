#ifndef POLARIS_SIM_WORLD_SRP_HPP
#define POLARIS_SIM_WORLD_SRP_HPP

/// @file
/// @brief Solar radiation pressure, cannon-ball model (REQ-SIM-002, §5.2).
///
/// Photon momentum transfer pushes the spacecraft directly away from the Sun:
///
///   a = nu * P_1AU * C_R * (A/m) * (AU/|d|)^2 * d_hat,   d = r_sat - r_sun,
///
/// with `nu` the conical shadow factor (`eclipse.hpp`), `C_R` the radiation-
/// pressure coefficient (1 fully absorbing, 2 fully reflecting, ~1.2–1.5 typical),
/// and `A/m` the area-to-mass ratio [m^2/kg]. The `(AU/|d|)^2` factor spreads the
/// 1 AU reference pressure over the actual Sun-spacecraft range.
///
/// This is the **cannon-ball** approximation: a sphere of fixed cross-section,
/// so the force does not depend on attitude. That is the standard orbit-
/// determination model and is what the acceleration path implements. Real
/// spacecraft are not spheres, so the *torque* path adds the one attitude effect
/// that actually matters for ADCS — an offset between the center of pressure and
/// the center of mass, which turns the SRP force into a disturbance torque
/// (REQ-SIM-002 names SRP among the required disturbance torques). With the
/// default zero offset the torque is zero and the model is pure cannon-ball.
///
/// Above roughly 800 km SRP overtakes drag as the dominant non-gravitational
/// perturbation, and it never decays with altitude the way drag does.
///
/// Sim-side (`sim/CLAUDE.md`): heap / std::function / virtual dispatch are fine.
///
/// References:
///  - Montenbruck & Gill, *Satellite Orbits*, 2000, §3.4 (SRP, shadow).
///    [montenbruck2000]
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §8.6.4.
///    [vallado2013]

#include <Eigen/Core>
#include <utility>

#include "dynamics/force_torque.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "state/truth_state.hpp"
#include "world/body_position.hpp"

namespace polaris::sim::world {

/// Cannon-ball solar radiation pressure with a conical eclipse.
class SolarRadiationPressure : public dynamics::ForceTorqueModel {
 public:
  /// Area and mass are taken separately rather than as the usual lumped A/m
  /// because the disturbance torque needs the actual force, and force needs the
  /// mass that the acceleration path divides back out.
  ///
  /// @param area Cross-sectional area A [m^2].
  /// @param mass Spacecraft mass m [kg] (must be > 0).
  /// @param cr   Radiation-pressure coefficient C_R [-] (1 absorbing,
  ///             2 specularly reflecting).
  /// @param sun  Sun position resolver (`body_position.hpp`). Epochs it cannot
  ///             cover contribute no SRP.
  SolarRadiationPressure(double area, double mass, double cr, BodyPositionFn sun)
      : area_(area), mass_(mass), cr_(cr), sun_(std::move(sun)) {}

  /// Anti-sunward acceleration [m/s^2], scaled by the shadow factor.
  math::Vec3<math::frames::ECI> acceleration(const state::TruthState& s) const override;

  /// Disturbance torque tau = r_cp x F_body from the center-of-pressure offset.
  /// Zero unless `setCenterOfPressureOffset()` has been called.
  math::Vec3<math::frames::Body> torque(const state::TruthState& s) const override;

  /// Center of pressure relative to the center of mass, in Body [m]. A nonzero
  /// offset is what makes SRP an attitude disturbance rather than a pure force.
  void setCenterOfPressureOffset(const math::Vec3<math::frames::Body>& r_cp) { r_cp_ = r_cp; }

  /// Enable or disable the conical shadow. Default is enabled — eclipse is the
  /// physical behaviour, and disabling it is a deliberate analysis choice
  /// (isolating the secular SRP effect, or matching a reference tool run that
  /// models no shadow), not a fidelity shortcut.
  void setEclipseEnabled(bool enabled) { eclipse_enabled_ = enabled; }

  bool eclipseEnabled() const { return eclipse_enabled_; }

  /// The lumped orbit-determination parameter A/m [m^2/kg].
  double areaToMass() const { return area_ / mass_; }

  double cr() const { return cr_; }

 private:
  /// Shared core of `acceleration()`/`torque()`: the ECI SRP acceleration, or
  /// false if the Sun is uncovered at this epoch or the geometry is degenerate.
  bool solarAcceleration(const state::TruthState& s, Eigen::Vector3d& a_eci) const;

  double area_;
  double mass_;
  double cr_;
  BodyPositionFn sun_;
  math::Vec3<math::frames::Body> r_cp_{};  ///< default zero -> torque-free
  bool eclipse_enabled_{true};

  static constexpr double kMinDistance_ = 1.0;  ///< [m] singular-range guard
};

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_SRP_HPP
