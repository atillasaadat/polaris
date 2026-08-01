#ifndef POLARIS_SIM_WORLD_GRAVITY_GRADIENT_HPP
#define POLARIS_SIM_WORLD_GRAVITY_GRADIENT_HPP

/// @file
/// @brief Gravity-gradient disturbance torque (design doc §5.3; REQ-SIM-002).
///
/// The Earth's pull is not uniform across a finite body: the near side is pulled
/// harder than the far side, and unless the body is spherically symmetric the
/// resulting couple tries to align its **minimum-inertia axis with nadir**. That
/// is the whole of passive gravity-gradient stabilisation, and in LEO it is the
/// largest environmental torque on most spacecraft.
///
/// Contributes no acceleration: the point-mass force already lives in
/// `TwoBodyGravity` / `SphericalHarmonicGravity`, and the mass-distribution
/// correction to the *force* is smaller than this torque's effect by the ratio of
/// the body size to the orbit radius. Adding it here would also double-count the
/// central term.
///
/// Sim-side (`sim/CLAUDE.md`): virtual dispatch is fine.
///
/// References:
///  - Hughes, *Spacecraft Attitude Dynamics*, 1986, ch. 9 (gravitational torque,
///    libration). [hughes1986]
///  - Wertz (ed.), *Spacecraft Attitude Determination and Control*, 1978, §18.2.
///    [wertz1978]

#include <Eigen/Core>

#include "constants/constants.hpp"
#include "dynamics/force_torque.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "state/truth_state.hpp"

namespace polaris::sim::world {

/// Point-mass gravity-gradient torque on a rigid body.
///
/// **Model.** With \f$\hat{\mathbf r}\f$ the unit geocentric position vector
/// expressed in Body and \f$\mathbf J\f$ the body inertia tensor (Hughes ch. 9;
/// Wertz §18.2) [hughes1986; wertz1978]:
/// \f[
///   \boldsymbol\tau_{gg} = \frac{3\mu}{r^{3}}\;\hat{\mathbf r}\times(\mathbf J\,\hat{\mathbf r}).
/// \f]
/// \f$\hat{\mathbf r}\f$ enters twice, so nadir and anti-nadir give the same
/// torque — there is no sign convention to get wrong. The torque vanishes for a
/// spherical \f$\mathbf J\f$ and whenever \f$\hat{\mathbf r}\f$ lies along a
/// principal axis (the equilibria), and peaks at 45° between two principal axes.
///
/// This is the **degree-0** (point-mass) gradient, which is the committed
/// baseline: the J2 correction to the *gradient* is ~1e-3 of it in LEO (§5.3).
///
/// The inertia is referenced rather than copied purely so this model does not
/// own a second copy of it. That is **not** enough to make a time-varying
/// inertia work: `RigidBody6Dof` copies the tensor and precomputes its inverse
/// at construction, so mutating the referenced tensor mid-run would change this
/// torque while Euler's equation kept integrating the old one. §5.4's
/// propellant-depletion inertia change needs the plant and this provider to
/// share one tensor (and the plant to refresh its inverse); that is future work.
class GravityGradientTorque : public dynamics::ForceTorqueModel {
 public:
  /// @param inertia_kgm2 Body inertia tensor [kg·m²]. Referenced, not copied —
  ///        must outlive this model.
  /// @param mu          Gravitational parameter [m³/s²]; pass the gravity
  ///        model's own GM so the gradient and the force agree.
  explicit GravityGradientTorque(const Eigen::Matrix3d& inertia_kgm2,
                                 double mu = constants::wgs84::kGM)
      : inertia_(&inertia_kgm2), mu_(mu) {}

  /// A temporary would leave the held pointer dangling immediately.
  GravityGradientTorque(Eigen::Matrix3d&&, double = constants::wgs84::kGM) = delete;

  math::Vec3<math::frames::ECI> acceleration(const state::TruthState&) const override {
    return math::Vec3<math::frames::ECI>::Zero();
  }

  math::Vec3<math::frames::Body> torque(const state::TruthState& s) const override;

  double mu() const { return mu_; }

 private:
  const Eigen::Matrix3d* inertia_;
  double mu_;

  static constexpr double kMinRadius_ = 1.0;  ///< [m] singular-radius guard (§3.6)
};

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_GRAVITY_GRADIENT_HPP
