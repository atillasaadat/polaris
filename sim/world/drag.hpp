#ifndef POLARIS_SIM_WORLD_DRAG_HPP
#define POLARIS_SIM_WORLD_DRAG_HPP

/// @file
/// @brief Atmospheric drag, cannon-ball model (REQ-SIM-002, §5.2).
///
/// Momentum exchange with the neutral atmosphere, opposing the motion of the
/// spacecraft *through the air*:
///
///   a = -1/2 rho C_D (A/m) |v_rel| v_rel,   v_rel = v_eci - omega_e z_hat x r.
///
/// The `omega x r` term is the atmosphere co-rotating with the Earth, and it is
/// not a detail: at LEO it is ~460 m/s at the equator against a ~7.7 km/s orbital
/// speed, so it turns drag into a force with an out-of-plane component that
/// drives the secular inclination and node behaviour. Dropping it leaves a model
/// that still looks plausible and is quietly wrong (Vallado Eq. 8-32).
///
/// Density comes from an injected `DensityFn` (`atmosphere.hpp`), which is what
/// keeps this model independent of NRLMSIS and the space-weather files — see that
/// header. Drag is the perturbation with by far the largest model uncertainty:
/// thermospheric density swings by an order of magnitude over the solar cycle,
/// and `C_D` is an effective fitted parameter, not a measured one.
///
/// Like `SolarRadiationPressure`, the acceleration path is pure cannon-ball
/// (fixed cross-section, attitude-independent) and the torque path adds the one
/// attitude effect ADCS cares about: a center-of-pressure / center-of-mass offset
/// turns the drag force into the **aero** disturbance torque REQ-SIM-002 names.
/// With the default zero offset the torque is zero.
///
/// Sim-side (`sim/CLAUDE.md`): heap / std::function / virtual dispatch are fine.
///
/// References:
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
///    §8.6.2 (drag, co-rotating atmosphere). [vallado2013]
///  - Montenbruck & Gill, *Satellite Orbits*, 2000, §3.5 (atmospheric drag).
///    [montenbruck2000]

#include <Eigen/Core>
#include <utility>

#include "dynamics/force_torque.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "state/truth_state.hpp"
#include "world/atmosphere.hpp"

namespace polaris::sim::world {

/// Cannon-ball atmospheric drag against a co-rotating atmosphere.
///
/// **Model.** The acceleration opposes the atmosphere-relative velocity
/// (Vallado Eq. 8-32; Montenbruck & Gill §3.5) [vallado2013; montenbruck2000]:
/// \f[
///   \mathbf a = -\tfrac12\,\rho\,\frac{C_D A}{m}\,|\mathbf v_{\mathrm{rel}}|\,
///     \mathbf v_{\mathrm{rel}},
///   \qquad
///   \mathbf v_{\mathrm{rel}} = \mathbf v - \boldsymbol\omega_\oplus\times\mathbf r,
///   \quad \boldsymbol\omega_\oplus = \omega_\oplus\,\hat{\mathbf z},
/// \f]
/// where \f$\rho\f$ is the neutral density from the injected `DensityFn` and the
/// \f$\boldsymbol\omega_\oplus\times\mathbf r\f$ term is the co-rotating wind.
/// A center-of-pressure offset \f$\mathbf r_{cp}\f$ turns the force into the aero
/// disturbance torque \f$\boldsymbol\tau = \mathbf r_{cp}\times\mathbf F\f$ with
/// \f$\mathbf F = m\mathbf a\f$ resolved in Body; with the default zero offset the
/// torque vanishes.
class AtmosphericDrag : public dynamics::ForceTorqueModel {
 public:
  /// Area and mass are taken separately rather than as the usual lumped
  /// ballistic coefficient because the aero disturbance torque needs the actual
  /// force, and force needs the mass that the acceleration path divides out.
  ///
  /// @param area    Cross-sectional area A [m^2].
  /// @param mass    Spacecraft mass m [kg] (must be > 0).
  /// @param cd      Drag coefficient C_D [-] (~2.2 for a compact LEO body).
  /// @param density Neutral-density resolver (`atmosphere.hpp`). Altitudes it
  ///                reports as zero density contribute no drag.
  AtmosphericDrag(double area, double mass, double cd, DensityFn density)
      : area_(area), mass_(mass), cd_(cd), density_(std::move(density)) {}

  /// Acceleration [m/s^2], anti-parallel to the atmosphere-relative velocity.
  math::Vec3<math::frames::ECI> acceleration(const state::TruthState& s) const override;

  /// Aero disturbance torque tau = r_cp x F_body from the center-of-pressure
  /// offset. Zero unless `setCenterOfPressureOffset()` has been called.
  math::Vec3<math::frames::Body> torque(const state::TruthState& s) const override;

  /// Center of pressure relative to the center of mass, in Body [m]. A nonzero
  /// offset is what makes drag an attitude disturbance rather than a pure force.
  void setCenterOfPressureOffset(const math::Vec3<math::frames::Body>& r_cp) { r_cp_ = r_cp; }

  /// The lumped ballistic parameter C_D A/m [m^2/kg].
  double ballisticCoefficient() const { return cd_ * area_ / mass_; }

  double cd() const { return cd_; }

  /// Velocity relative to the co-rotating atmosphere [m/s] at truth state @p s.
  /// Exposed because it is the piece most worth checking independently.
  static Eigen::Vector3d relativeVelocity(const state::TruthState& s);

 private:
  /// Shared core of `acceleration()`/`torque()`: the ECI drag acceleration, or
  /// false if there is no atmosphere here or the geometry is degenerate.
  bool dragAcceleration(const state::TruthState& s, Eigen::Vector3d& a_eci) const;

  double area_;
  double mass_;
  double cd_;
  DensityFn density_;
  math::Vec3<math::frames::Body> r_cp_{};  ///< default zero -> torque-free

  static constexpr double kMinSpeed_ = 1.0e-6;  ///< [m/s] singular-direction guard
};

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_DRAG_HPP
